import os, sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from Dataset.utils import (
    self_recorded_to_common_voice_format_tsv,
    preprocess_common_voice_tsvs,
    preprocess_openSLR,
    Hi2EngTransliterator,
)
from configs.config import (
    self_recorded_audio_clips_dir,
    self_recorded_transcript_txt,
    common_voice_hi_root,
    common_voice_en_root,
    openSLR_train_transcripts,
    openSLR_train_processed_transcript_tsv,
    openSLR_test_transcripts,
    openSLR_test_processed_transcript_tsv,
)

if __name__ == "__main__":
    transliterator = Hi2EngTransliterator()
    # self_recorded_to_common_voice_format_tsv(
    #     self_recorded_audio_clips_dir.resolve().as_posix(),
    #     self_recorded_transcript_txt.resolve().as_posix(),
    #     self_recorded_audio_clips_dir.parent.resolve().as_posix(),
    #     audio_file_path_prefix="Recording ",
    #     test_size=0.2,
    # )
    # preprocess_common_voice_tsvs(
    #     common_voice_dir_path=common_voice_hi_root.as_posix(),
    #     save_dir=(common_voice_hi_root / "processed").as_posix(),
    #     transliterator=transliterator,
    #     cache=True,
    # )
    # No need to preprocess common_voice en tsv
    preprocess_openSLR(
        openSLR_train_transcripts.absolute().as_posix(),
        openSLR_train_processed_transcript_tsv.absolute().as_posix(),
        transliterator=transliterator,
    )
    preprocess_openSLR(
        openSLR_test_transcripts.absolute().as_posix(),
        openSLR_test_processed_transcript_tsv.absolute().as_posix(),
        transliterator=transliterator,
    )
