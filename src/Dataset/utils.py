import torchaudio
import torch
from typing import Generator, Tuple, Dict


def load_audio_file(file_path: str, mono: bool = True) -> Tuple[torch.Tensor, int]:
    y, sr = torchaudio.load(file_path)  # type: ignore
    if mono:
        y = torch.mean(y, dim=0).squeeze(0)
    return y, sr


def get_recording_number(file_name: str, prefix: str = "Recording "):
    return int(file_name.replace(prefix, "").split(".")[0])


def chunk_processed_inputs(
    input_dict, chunkby: int
) -> Generator[Dict[str, torch.Tensor], None, None]:
    """input_dict is the dict out of a processor. It must have
    `input_values` and `attention_mask`
    """
    assert "input_values" in input_dict
    assert "attention_mask" in input_dict
    input_values = input_dict["input_values"]
    attention_mask = input_dict["attention_mask"]

    input_size = input_values.size(1)
    num_sections = input_size // chunkby

    if not num_sections:
        yield {"input_values": input_values, "attention_mask": attention_mask}

    input_values_split = torch.tensor_split(input_values, num_sections, dim=1)
    attention_mask_split = torch.tensor_split(attention_mask, num_sections, dim=1)

    for i, a in zip(input_values_split, attention_mask_split):
        yield {"input_values": i, "attention_mask": a}
