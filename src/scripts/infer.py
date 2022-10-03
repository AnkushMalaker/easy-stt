import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torchaudio.functional as taF
from Dataset import load_audio_file
from Models import AvailablePretrainedModels, load_model
from typing import Optional, List
from argparse import ArgumentParser
import pandas as pd


def infer_from_sample(file_path, model, processor, device) -> List[str]:
    resampled_rate = processor.feature_extractor.sampling_rate  # type: ignore
    y_mono, orig_sr = load_audio_file(file_path, mono=True)
    resampled_y_mono = taF.resample(y_mono, orig_sr, resampled_rate)
    inputs = processor(resampled_y_mono, sampling_rate=resampled_rate, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs.to(device)).logits
        predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids.cpu())
    return transcription


def main(
    input_file_or_dir_path: str,
    model_name: str,
    output_file_path: Optional[str] = None,
    device: str = "cuda",
    weights_path: Optional[str] = None,
):
    DEVICE = torch.device(device)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    if weights_path:
        if os.path.isdir(weights_path):
            model = load_model(weights_path)
            print(f"Loaded weights from {weights_path}")
        else:
            model = load_model(model_name)
    else:
        model = load_model(model_name)
    model = model.to(DEVICE)

    transcripts = []
    if os.path.isfile(input_file_or_dir_path):
        transcript = infer_from_sample(input_file_or_dir_path, model, processor, DEVICE)
        transcripts = [{"filename": input_file_or_dir_path, "transcript": transcript[0]}]
        print(transcript[0])
    else:
        file_names = os.listdir(input_file_or_dir_path)
        file_paths = [os.path.join(input_file_or_dir_path, file_name) for file_name in file_names]

        for file_name, file_path in zip(file_names, file_paths):

            transcription = infer_from_sample(file_path, model, processor, device=DEVICE)
            transcripts.append({"filename": file_name, "transcript": transcription[0]})

            print(file_name)
            print(transcription[0])
            print()

    if output_file_path:
        df = pd.DataFrame(transcripts, index=[0] if len(transcripts) == 1 else None)
        df.set_index("filename", inplace=True)
        df.to_csv(output_file_path, sep=",")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file_or_dir_path", type=str)
    parser.add_argument("output_file_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--choose_model", "-c", default=False, action="store_true")
    parser.add_argument("--weights", "-w", type=str)
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

    input_file_or_dir_path = args.input_file_or_dir_path
    output_file_path = args.output_file_path
    device = args.device
    weights_path = args.weights

    main(
        input_file_or_dir_path=input_file_or_dir_path,
        output_file_path=output_file_path,
        model_name=model_name,
        device=device,
        weights_path=weights_path,
    )
