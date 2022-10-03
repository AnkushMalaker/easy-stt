from torch.utils.data import Dataset
import math
import torch
from dataclasses import dataclass

from torch.utils.data.dataset import ConcatDataset
from .utils import (
    common_voice_tsv_to_dict,
    load_audio_file,
    DatasetConf,
    openslr_to_dict,
    sample_dataset_dict,
)
from transformers.models.wav2vec2.processing_wav2vec2 import Wav2Vec2Processor
from typing import Dict, List, Optional, Union
import torchaudio.functional as taF
import os

# Procecss common voice

# FIXME: Add step in preprocessing where training sample is split it two if too big?
# How would you assign labels then tho sheesh


class ASRDataset(Dataset):
    def __init__(
        self,
        data_dict: Dict[str, str],
        processor: Wav2Vec2Processor,
        max_processed_audio_len: int = 16000 * 30,
    ) -> None:
        super().__init__()

        self.resampling_rate = processor.feature_extractor.sampling_rate  # type: ignore
        self.processor = processor

        self.samples_list = list(data_dict.items())  # list of tuples (fpath, text)

        self.max_processed_audio_len = max_processed_audio_len

    def load_sample(self, audio_filepath: str, text: str):
        audio_array, orig_sr = load_audio_file(filepath=audio_filepath, mono=True)

        audio_array = taF.resample(audio_array, orig_sr, self.resampling_rate)

        processed_audio = self.processor(audio=audio_array, sampling_rate=self.resampling_rate)[
            "input_values"
        ][0]
        original_processed_audio_len = processed_audio.size  # These are numpy arrays so size is int
        processed_audio = processed_audio[: self.max_processed_audio_len]
        ratio = processed_audio.size / original_processed_audio_len

        trimmed_text = text.split(" ")
        trimmed_text = trimmed_text[0 : math.ceil(len(trimmed_text) * ratio)]
        labels = self.processor(text=" ".join(trimmed_text)).input_ids

        return {"labels": labels, "input_values": processed_audio}

    def __len__(self) -> int:
        return len(self.samples_list)

    def __getitem__(self, index):
        return self.load_sample(*self.samples_list[index])


def load_dataset(
    dataset_conf: DatasetConf, processor, max_processed_audio_len: int = 16000 * 30
) -> ASRDataset:
    print(f"Loading {dataset_conf.labels_file}")
    # audio_clips_dir_path: Union[str, Path]
    # labels_file: Union[str, Path]
    # type: Optional[str] = None
    if dataset_conf.dataset_type == "OpenSLR":
        dataset_dict = openslr_to_dict(
            audio_clips_dir_path=dataset_conf.audio_clips_dir_path,
            transcription_filepath=dataset_conf.labels_file,
        )
        asr_dataset = ASRDataset(
            dataset_dict, processor=processor, max_processed_audio_len=max_processed_audio_len
        )
        print(asr_dataset.__len__())
        return asr_dataset
    elif dataset_conf.dataset_type == "CommonVoice":
        dataset_dict = common_voice_tsv_to_dict(
            clips_dir_path=dataset_conf.audio_clips_dir_path, csv_path=dataset_conf.labels_file
        )
        dataset_dict = sample_dataset_dict(dataset_dict, dataset_conf.sample_size)
        asr_dataset = ASRDataset(
            dataset_dict, processor=processor, max_processed_audio_len=max_processed_audio_len
        )
        print(asr_dataset.__len__())
        return asr_dataset
    else:
        print("Unimplimented")
        raise NotImplementedError


def combine_datasets(
    dataset_conf_list: List[DatasetConf], processor, max_processed_audio_len: int = 16000 * 30
):
    return ConcatDataset(
        [
            load_dataset(
                ds_config, processor=processor, max_processed_audio_len=max_processed_audio_len
            )
            for ds_config in dataset_conf_list
        ]
    )


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
