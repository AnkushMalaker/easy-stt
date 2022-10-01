import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import torch
from transformers import Wav2Vec2Processor
import torchaudio.functional as taF
from Dataset import load_audio_file, chunk_processed_inputs
from Models import AvailablePretrainedModels, load_model
from typing import Optional
from argparse import ArgumentParser

MUL_FOR_16K = 18666.666666666668  # idk how this comes, I manually checked this.
CHUNK_SECONDS = 20


def main(
    input_file_path: str,
    output_file_path: str,
    model_name: str,
    device: str,
    weights_path: Optional[str] = None,
):
    DEVICE = torch.device(device)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    if weights_path:
        if os.path.isfile(weights_path):
            model = load_model(weights_path)
        else:
            model = load_model(model_name)
    else:
        model = load_model(model_name)
    model = model.to(DEVICE)

    resampled_rate = processor.feature_extractor.sampling_rate  # type: ignore

    y_mono, orig_sr = load_audio_file(input_file_path, mono=True)
    resampled_y_mono = taF.resample(y_mono, orig_sr, resampled_rate)
    inputs = processor(resampled_y_mono, sampling_rate=resampled_rate, return_tensors="pt")
    transcriptions = []
    with torch.no_grad():
        for input in chunk_processed_inputs(inputs, chunkby=int(CHUNK_SECONDS * MUL_FOR_16K)):
            input_values = input["input_values"].to(DEVICE)
            attention_mask = input["attention_mask"].to(DEVICE)
            logits = model(input_values=input_values, attention_mask=attention_mask).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            transcription = processor.batch_decode(predicted_ids.cpu())
            transcriptions.append(transcription[0])

    with open(output_file_path, "w") as f:
        f.writelines("\n".join(transcriptions))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file_path", type=str)
    parser.add_argument("output_file_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--device", "-d", type=str, default="cuda")
    parser.add_argument("--choose_model", "-c", default=False, action="store_true")
    parser.add_argument("--weights", "-w", type=str)
    args = parser.parse_args()

    model_name = args.model_name
    if args.choose_model:
        models = AvailablePretrainedModels.models
        print("Avaliable Models:")
        for i, m_name in enumerate(models):
            print(i + 1, ": ", m_name)
        print("Enter choice: ")
        choice = int(input())
        model_name = models[choice - 1]
        if args.model_name:
            print(f"Using {model_name}")

    device = args.device

    weights_path = args.weights
    input_file_path: str = args.input_file_path
    output_file_path: str = args.output_file_path
    assert input_file_path.endswith("wav")
    assert output_file_path.endswith("txt")

    main(input_file_path, output_file_path, model_name, device, weights_path=weights_path)
