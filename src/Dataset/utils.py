from functools import cache
import time
import torchaudio
import tqdm
from pathlib import Path
from dataclasses import dataclass
import random
import torch
import os
from typing import Generator, Optional, Tuple, Dict, cast, List, Union
import pandas as pd

from ai4bharat.transliteration import XlitEngine


@dataclass
class DatasetConf:
    audio_clips_dir_path: str
    labels_file: str
    dataset_type: Optional[str] = None
    sample_size: Optional[Union[float, int]] = None


class Hi2EngTransliterator:
    """
    Using cache makes it easy to translit corpuses with many repeated labels.
    When processing multiple files, use one instance of this transliterator so that
    caching can be more effective
    """

    def __init__(self, beam_width: int = 10, rescore: bool = True) -> None:
        self.transliterator = XlitEngine(
            lang2use="hi", src_script_type="indic", beam_width=beam_width, rescore=rescore
        )

    @cache
    def _transliterate_sentence_cache(self, sentence: str) -> str:
        # When initializing XlitEngine with single lang2use, returns str
        return self.transliterator.translit_sentence(sentence, lang_code="hi")  # type: ignore

    def _transliterate_sentence_no_cache(self, sentence: str) -> str:
        # When initializing XlitEngine with single lang2use, returns str
        return self.transliterator.translit_sentence(sentence, lang_code="hi")  # type: ignore

    def transliterate_sentence(self, sentence: str, cache: bool = True) -> str:
        # When initializing XlitEngine with single lang2use, returns str
        if cache:
            return self._transliterate_sentence_cache(sentence)
        else:
            return self._transliterate_sentence_no_cache(sentence)

    # Effect of using cache
    # Finished transliteration on other.tsv in 11.18342890739441 minutes.
    # Finished transliteration on validated.tsv in 20.46280355056127 minutes.
    # Finished transliteration on invalidated.tsv in 1.9321432709693909 minutes.
    # Finished transliteration on test.tsv in 4.380544026692708e-05 minutes.
    # Finished transliteration on train.tsv in 6.40551249186198e-05 minutes.
    # Finished transliteration on reported.tsv in 0.0723628838857015 minutes.


def preprocess_common_voice_tsv(
    tsv_path: str,
    save_path: Optional[str] = None,
    transliterator: Optional[Hi2EngTransliterator] = None,
    cache: bool = True,
):

    from tqdm import tqdm

    tqdm.pandas()
    df = cast(pd.DataFrame, pd.read_csv(tsv_path, sep="\t"))
    if transliterator is None:
        transliterator = Hi2EngTransliterator()
    keep_columns = ["sentence", "path"]

    drop_columns = []
    for col in df.columns.values:
        if col not in keep_columns:
            drop_columns.append(col)

    df.drop(drop_columns, axis=1, inplace=True)
    st = time.time()
    df["sentence"] = df["sentence"].progress_apply(
        transliterator.transliterate_sentence, cache=cache
    )
    print(f"Finished transliteration on {tsv_path} in {(time.time() - st)/60} minutes.")
    if save_path is None:
        # Over write original
        df.to_csv(tsv_path, sep="\t", index=False)
    else:
        df.to_csv(save_path, sep="\t", index=False)
        print(df.head().to_string())


def preprocess_openSLR(
    transcript_txt_path: str,
    openSLR_transcript_tsv_save_path: str,
    transliterator: Hi2EngTransliterator,
    cache: bool = True,
):
    def process_line(line) -> Tuple[str, str]:
        split_line = line.split(" ")
        file_name = split_line[0]
        text = " ".join(split_line[1:])
        return file_name, text

    with open(transcript_txt_path, "r") as f:
        lines = f.readlines()

    filename_text_dict = {}
    for line in tqdm.tqdm(lines):
        filename, text = process_line(line)
        filename = filename + ".wav"

        text = transliterator.transliterate_sentence(text, cache)
        filename_text_dict[filename] = text

    filename_text_list = list(filename_text_dict.items())

    df = pd.DataFrame(filename_text_list, columns=["path", "sentence"])

    df.to_csv(openSLR_transcript_tsv_save_path, sep="\t", index=False)


def preprocess_common_voice_tsvs(
    common_voice_dir_path,
    save_dir: Optional[str] = None,
    transliterator: Optional[Hi2EngTransliterator] = None,
    cache: bool = True,
):
    filenames = [
        x
        for x in os.listdir(common_voice_dir_path)
        if os.path.isfile(os.path.join(common_voice_dir_path, x)) and x.endswith(".tsv")
    ]
    print(os.listdir(common_voice_dir_path))
    filepaths = [os.path.join(common_voice_dir_path, filename) for filename in filenames]
    if save_dir is None:
        savepaths = filepaths
    else:
        os.makedirs(save_dir, exist_ok=True)
        savepaths = [os.path.join(save_dir, filename) for filename in filenames]

    total = len(filepaths)
    for filepath, savepath in tqdm.tqdm(zip(filepaths, savepaths), total=total):
        print(filepath, savepath)
        preprocess_common_voice_tsv(filepath, savepath, transliterator, cache=cache)


def common_voice_tsv_to_dict(
    clips_dir_path,
    csv_path,
) -> Dict[str, str]:
    df = cast(pd.DataFrame, pd.read_csv(csv_path, sep="\t"))
    assert "sentence" in df
    filenames = list(df["path"])
    sentences: List[str] = list(df["sentence"])

    filepaths = [os.path.join(clips_dir_path, filename) for filename in filenames]

    filepath_sentence_dict = {
        filepath: sentence for filepath, sentence in zip(filepaths, sentences)
    }

    return filepath_sentence_dict


def openslr_to_dict(audio_clips_dir_path: str, transcription_filepath: str):
    def process_line(line) -> Tuple[str, str]:
        split_line = line.split(" ")
        file_name = split_line[0]
        text = " ".join(split_line[1:])
        return file_name, text

    with open(transcription_filepath, "r") as f:
        lines = f.readlines()

    filepaths_text_dict = {}
    for line in lines:
        filename, text = process_line(line)
        filepath = os.path.join(audio_clips_dir_path, filename + ".wav")
        filepaths_text_dict[filepath] = text

    return filepaths_text_dict


# def split_self_record_into_train_test(
#     audio_clips_dir_path: str,
#     transcription_filepath: str,
#     test_split_percentage: float = 0.2,
#     prefix="Recording ",
# ):
#     samples = self_recorded_audio_dataset_to_dict(
#         audio_clips_dir_path, transcription_filepath, audio_file_path_prefix=prefix
#     )

#     # for sample in train_list:


def self_recorded_to_common_voice_format_tsv(
    audio_clips_dir_path: str,
    sample_text_path: str,
    tsv_save_dir_path: str,
    audio_file_path_prefix: str = "Recording ",
    test_size: float = 0.2,
):
    samples_dict = _preprocessing_self_recorded_audio_dataset_to_dict(
        audio_clips_dir_path, sample_text_path, audio_file_path_prefix=audio_file_path_prefix
    )

    COLUMNS = [
        "path",
        "sentence",
    ]

    samples_list = list(samples_dict.items())
    random.shuffle(samples_list)

    total = len(samples_list)
    train_samples = int(total - total * test_size)

    df_train = pd.DataFrame(samples_list[0:train_samples], columns=COLUMNS)
    df_test = pd.DataFrame(samples_list[train_samples:], columns=COLUMNS)

    df_train.to_csv(Path(tsv_save_dir_path) / "train.tsv", sep="\t", index=False)
    df_test.to_csv(Path(tsv_save_dir_path) / "test.tsv", sep="\t", index=False)


def _preprocessing_self_recorded_audio_dataset_to_dict(
    audio_dir_path,
    true_transcripts_txt_file_path: str,
    audio_file_path_prefix: str = "Recording ",
) -> Dict[str, str]:

    with open(true_transcripts_txt_file_path, "r") as f:
        transcripts = f.readlines()
    transcripts = [l.strip() for l in transcripts]

    audio_file_names = os.listdir(audio_dir_path)
    audio_file_paths = [
        os.path.join(audio_dir_path, audio_file_name) for audio_file_name in audio_file_names
    ]
    samples = {}
    for audio_filename, audio_file_path in zip(audio_file_names, audio_file_paths):
        recording_id = get_recording_number(file_name=audio_filename, prefix=audio_file_path_prefix)
        transcript_id = recording_id - 1  # Assuming recordings filenames start from 1
        samples[audio_file_path] = transcripts[transcript_id]
    return samples


def sample_dataset_dict(
    dataset_dict: Dict[str, str], sample_size: Optional[Union[float, int]]
) -> Dict[str, str]:
    if sample_size is None or (isinstance(sample_size, float) and sample_size == 1.0):
        return dataset_dict
    elif isinstance(sample_size, float) and sample_size < 1.0:
        dataset_list = list(dataset_dict.items())
        random.shuffle(dataset_list)
        dataset_list = dataset_list[0 : int(len(dataset_list) * sample_size)]
        return dict(dataset_list)
    elif isinstance(sample_size, int):
        dataset_list = list(dataset_dict.items())
        random.shuffle(dataset_list)
        dataset_list = dataset_list[0:sample_size]
        return dict(dataset_list)
    else:
        print("Couldn't sample from dataset_dict. Using full dataset.")
        return dataset_dict


def load_audio_file(filepath: str, mono: bool = True) -> Tuple[torch.Tensor, int]:
    y, sr = torchaudio.load(filepath)  # type: ignore
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
