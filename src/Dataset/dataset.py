from torch.utils.data import Dataset
import torch
from dataclasses import dataclass
from .utils import get_recording_number, load_audio_file
from transformers import Wav2Vec2Processor
from typing import Dict, List, Optional, Union
import torchaudio.functional as taF
import os


class ASRDataset(Dataset):
    def __init__(
        self,
        audio_dir_path: str,
        true_transcripts_txt_file_path: str,
        processor: Wav2Vec2Processor,
        audio_file_path_prefix: str = "Recording ",
    ) -> None:
        super().__init__()

        with open(true_transcripts_txt_file_path, "r") as f:
            transcripts = f.readlines()
        self.transcripts = [l.strip() for l in transcripts]

        audio_file_names = os.listdir(audio_dir_path)
        audio_file_paths = [
            os.path.join(audio_dir_path, audio_file_name) for audio_file_name in audio_file_names
        ]
        self.samples = []
        resampling_rate = processor.feature_extractor.sampling_rate  # type: ignore
        for audio_filename, audio_file_path in zip(audio_file_names, audio_file_paths):
            recording_id = get_recording_number(
                file_name=audio_filename, prefix=audio_file_path_prefix
            )
            transcript_id = recording_id - 1  # Assuming recordings filenames start from 1
            audio_array, orig_sr = load_audio_file(file_path=audio_file_path, mono=True)

            audio_array = taF.resample(audio_array, orig_sr, resampling_rate)

            processed_audio = processor(audio=audio_array, sampling_rate=resampling_rate)[
                "input_values"
            ][0]
            labels = processor(text=self.transcripts[transcript_id]).input_ids

            self.samples.append({"labels": labels, "input_values": processed_audio})

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        max_length_labels (:obj:`int`, `optional`):
            Maximum length of the ``labels`` returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods

        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch
