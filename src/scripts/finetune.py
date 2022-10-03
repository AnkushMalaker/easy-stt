import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from Dataset import DataCollatorCTCWithPadding, combine_datasets
from configs import TrainDatasetConfigs, TestDatasetConfigs

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, TrainingArguments, Trainer

from datasets import load_metric
from argparse import ArgumentParser
import numpy as np


def main(
    pretrained_model_name: str,
    output_path: str,
    freeze_feature_encoder: bool = True,
):

    processor = Wav2Vec2Processor.from_pretrained(pretrained_model_name)
    wer_metric = load_metric("wer")

    model = Wav2Vec2ForCTC.from_pretrained(
        pretrained_model_name,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    data_collator = DataCollatorCTCWithPadding(processor, padding=True)

    if freeze_feature_encoder:
        model.freeze_feature_encoder()

    training_args = TrainingArguments(
        output_dir=output_path,
        group_by_length=True,
        per_device_train_batch_size=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        save_steps=1000,
        eval_steps=2000,
        logging_steps=100,
        learning_rate=1e-5,
        weight_decay=0.005,
        warmup_steps=1000,
        save_total_limit=2,
    )

    train_dataset = combine_datasets(TrainDatasetConfigs, processor=processor)
    test_dataset = combine_datasets(TestDatasetConfigs, processor=processor)

    def compute_wer_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        wer = wer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=processor.feature_extractor,
        compute_metrics=compute_wer_metrics,
    )

    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="One of the models specified in configs. Default: 'facebook/wav2vec2-base-960h'",
        default="facebook/wav2vec2-base-960h",
    )
    parser.add_argument(
        "--no_freeze_feature_encoder",
        action="store_false",
        help="Specify this flag if you'd like to train the whole model and not just the final layer.",
        default=True,
    )
    parser.add_argument("--output_path", "-o", default="./output_dir")
    args = parser.parse_args()
    model_name = args.model_name
    freeze_feature_encoder = args.no_freeze_feature_encoder
    output_path = args.output_path
    main(
        pretrained_model_name=model_name,
        output_path=output_path,
        freeze_feature_encoder=freeze_feature_encoder,
    )
