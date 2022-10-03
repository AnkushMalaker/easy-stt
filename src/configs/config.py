from pathlib import Path
from Dataset.utils import DatasetConf

home = str(Path.home())

# ProcessingFunctions = []


DATA_DIR = Path(f"{home}") / "Projects" / "custom_stt" / "Data"

# COMMON_VOICE_HI
common_voice_hi_root = DATA_DIR / "cv-corpus-11.0-2022-09-21-hi" / "hi"
common_voice_hi_clips_dir_path = common_voice_hi_root / "clips"

common_voice_hi_train_tsv = common_voice_hi_root / "train.tsv"
common_voice_hi_test_tsv = common_voice_hi_root / "test.tsv"
common_voice_hi_validated_tsv = common_voice_hi_root / "validated.tsv"
common_voice_hi_reported_tsv = common_voice_hi_root / "reported.tsv"
common_voice_hi_invalidated_tsv = common_voice_hi_root / "invalidated.tsv"
common_voice_hi_other_tsv = common_voice_hi_root / "other.tsv"


common_voice_hi_train_processed_tsv = common_voice_hi_root / "processed" / "train.tsv"
common_voice_hi_test_processed_tsv = common_voice_hi_root / "processed" / "test.tsv"
common_voice_hi_validated_processed_tsv = common_voice_hi_root / "processed" / "validated.tsv"
common_voice_hi_reported_processed_tsv = common_voice_hi_root / "processed" / "reported.tsv"
common_voice_hi_invalidated_processed_tsv = common_voice_hi_root / "processed" / "invalidated.tsv"
common_voice_hi_other_processed_tsv = common_voice_hi_root / "processed" / "other.tsv"

# COMMON_VOICE_EN
common_voice_en_root = DATA_DIR / "cv-corpus-11.0-2022-09-21-en" / "en"
common_voice_en_clips_dir_path = common_voice_en_root / "clips"

common_voice_en_train_tsv = common_voice_en_root / "train.tsv"
common_voice_en_test_tsv = common_voice_en_root / "test.tsv"
common_voice_en_validated_tsv = common_voice_en_root / "validated.tsv"
common_voice_en_reported_tsv = common_voice_en_root / "reported.tsv"
common_voice_en_invalidated_tsv = common_voice_en_root / "invalidated.tsv"
common_voice_en_other_tsv = common_voice_en_root / "other.tsv"

openSLR_root = DATA_DIR / "openSLR"

openSLR_train_audio_clips_dir = openSLR_root / "train" / "audio"
openSLR_train_transcripts = openSLR_root / "train" / "transcription.txt"
openSLR_train_processed_transcript_tsv = openSLR_root / "train" / "transcription_processed.tsv"

openSLR_test_audio_clips_dir = openSLR_root / "test" / "audio"
openSLR_test_transcripts = openSLR_root / "test" / "transcription.txt"
openSLR_test_processed_transcript_tsv = openSLR_root / "test" / "transcription_processed.tsv"

self_recorded_audio_clips_dir = DATA_DIR / "self_recorded" / "samples"
self_recorded_transcript_txt = DATA_DIR / "self_recorded" / "sample_text.txt"
self_recorded_transcript_train_tsv = DATA_DIR / "self_recorded" / "train.tsv"
self_recorded_transcript_test_tsv = DATA_DIR / "self_recorded" / "test.tsv"

PREDEFINED_DATASET_CONFS = {
    "CommonVoiceHiTrain": DatasetConf(
        common_voice_hi_clips_dir_path.resolve().as_posix(),
        common_voice_hi_train_processed_tsv.resolve().as_posix(),
        "CommonVoice",
    ),
    "CommonVoiceHiTest": DatasetConf(
        common_voice_hi_clips_dir_path.resolve().as_posix(),
        common_voice_hi_test_processed_tsv.resolve().as_posix(),
        "CommonVoice",
    ),
    "CommonVoiceEnTrain": DatasetConf(
        common_voice_en_clips_dir_path.resolve().as_posix(),
        common_voice_en_train_tsv.resolve().as_posix(),
        "CommonVoice",
        0.2,
    ),
    "CommonVoiceEnTest": DatasetConf(
        common_voice_en_clips_dir_path.resolve().as_posix(),
        common_voice_en_test_tsv.resolve().as_posix(),
        "CommonVoice",
        0.2,
    ),
    "OpenSLR_Train": DatasetConf(
        openSLR_train_audio_clips_dir.resolve().as_posix(),
        openSLR_train_processed_transcript_tsv.resolve().as_posix(),
        "CommonVoice",
    ),
    "OpenSLR_Test": DatasetConf(
        openSLR_test_audio_clips_dir.resolve().as_posix(),
        openSLR_test_processed_transcript_tsv.resolve().as_posix(),
        "CommonVoice",
    ),
    "SelfRecordedTrain": DatasetConf(
        self_recorded_audio_clips_dir.resolve().as_posix(),
        self_recorded_transcript_train_tsv.resolve().as_posix(),
        "CommonVoice",
    ),
    "SelfRecordedTest": DatasetConf(
        self_recorded_audio_clips_dir.resolve().as_posix(),
        self_recorded_transcript_test_tsv.resolve().as_posix(),
        "CommonVoice",
    ),
}

TrainDatasetConfigs = [
    PREDEFINED_DATASET_CONFS["CommonVoiceEnTrain"],
    PREDEFINED_DATASET_CONFS["CommonVoiceHiTrain"],
    PREDEFINED_DATASET_CONFS["OpenSLR_Train"],
    PREDEFINED_DATASET_CONFS["SelfRecordedTrain"],
]
TestDatasetConfigs = [
    PREDEFINED_DATASET_CONFS["CommonVoiceEnTest"],
    PREDEFINED_DATASET_CONFS["CommonVoiceHiTest"],
    PREDEFINED_DATASET_CONFS["OpenSLR_Test"],
    PREDEFINED_DATASET_CONFS["SelfRecordedTest"],
]
