import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
import io
import pyaudio
import soundfile as sf
from typing import List, Optional
from transformers import Wav2Vec2Processor, Wav2Vec2ConformerForCTC
import torchaudio.functional as taF
import torch.nn as nn
from Models import AvailablePretrainedModels, load_model
from argparse import ArgumentParser

SAMPLE_RATE = 47800


def write_wav(buffer, path, rate):
    sf.write(path, buffer, rate)


def read_audio(audio_device_index: int):
    CHUNK = 1024
    FORMAT = pyaudio.paFloat32
    CHANNELS = 1
    RATE = SAMPLE_RATE
    RECORD_SECONDS = 15

    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        input_device_index=audio_device_index,
    )

    print("Listening... ")
    print("press ctrl + c to stop")

    num_frames_to_fetch = int(RATE / CHUNK * RECORD_SECONDS)
    frames: List[bytes] = []
    while True:
        stream.start_stream()
        for _ in range(0, num_frames_to_fetch):
            data = stream.read(CHUNK)
            frames.append(data)
        # print("yeilding.")
        stream.stop_stream()
        yield frames
        # write_wav(np.frombuffer(b"".join(frames), np.float32), "/tmp/tmp_wav.wav", rate=47800)
        frames.clear()
    # FIXME: Check if need to close these resources
    # Not sure how to exit a generator properly
    # stream.stop_stream()
    # stream.close()
    # p.terminate()


def main(model_name: str, device: str, device_index: int, output_path: Optional[str] = None):
    DEVICE = torch.device(device)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model: nn.Module = load_model(model_name)
    model = model.to(DEVICE)

    resampled_rate = processor.feature_extractor.sampling_rate  # type: ignore

    transcriptions = []
    for frames in read_audio(device_index):

        combined_frames = b"".join(frames)

        data, _ = sf.read(
            io.BytesIO(combined_frames),
            channels=1,
            samplerate=SAMPLE_RATE,
            subtype="FLOAT",
            format="RAW",
        )

        resampled_y_mono = taF.resample(
            torch.tensor(data, dtype=torch.float32), SAMPLE_RATE, resampled_rate
        )
        inputs = processor(resampled_y_mono, sampling_rate=resampled_rate, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs.to(DEVICE)).logits
            predicted_ids = torch.argmax(logits, dim=-1)

        transcription = processor.batch_decode(predicted_ids.cpu())
        print(transcription[0], end=" ", flush=True)
        transcriptions.append(transcription[0])

    print("Finished. Cleaning Up.")
    if output_path:
        with open(output_path, "w") as f:
            f.writelines("\n".join(transcriptions))


def choose_audio_device() -> int:
    p = pyaudio.PyAudio()
    j = p.get_device_count()
    for i in range(j):
        print(f"Index: {i}: {p.get_device_info_by_index(i)['name']}")
    chosen_index = int(input())
    return chosen_index


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str)
    parser.add_argument(
        "--choose_model",
        "-m",
        default=False,
        help="specify this flag to choose one of available models via a prmopt shown.",
        action="store_true",
    )
    parser.add_argument(
        "--choose_audio_device",
        "-d",
        default=False,
        help="specify this flag to choose one of available audio_devices via a prmopt shown.",
        action="store_true",
    )
    args = parser.parse_args()
    if args.model_name:
        model_name = args.model_name
    elif args.choose_model:
        models = AvailablePretrainedModels.models
        print("Avaliable Models:")
        for i, m_name in enumerate(models):
            print(i + 1, ": ", m_name)
        print("Enter choice: ")
        choice = int(input())
        model_name = models[choice - 1]
        if args.model_name:
            print(f"Using {model_name}")
    else:
        print("Need to provide a model_name. Use -c to choose.")
        exit()

    if args.choose_audio_device:
        device_index = choose_audio_device()
    else:
        device_index = 0
    device = args.device
    main(model_name, device, device_index=device_index, output_path=args.output_path)
