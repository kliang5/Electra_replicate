import random
import evaluate
import numpy as np
from datasets import load_dataset
import torch
import numpy as np
from functools import partial
from transformers import (
    Trainer,
    ElectraConfig,
    ElectraTokenizerFast,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    TrainingArguments,
    default_data_collator,
)
from data import Glue
from model import Electra

SEED = 203


def pretrain(task_name):
    # Dataset
    raw_datasets = load_dataset(
        task_name,
    )

    # Labels
    is_regression = task_name == "stsb"
    if not is_regression:
        label_list = raw_datasets["train"].features["label"].names
        num_labels = len(label_list)
    else:
        num_labels = 1

    partial_func = partial(Glue.preprocess_function, task_name=task_name)
    raw_datasets = raw_datasets.map(
        partial_func,
        batched=True,
        desc="Running tokenizer on dataset",
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets[
        "validation_matched" if task_name == "mnli" else "validation"
    ]
    test_dataset = raw_datasets["test_matched" if task_name == "mnli" else "test"]

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Metric
    metric = evaluate.load("glue", task_name)

    # Tokenizer
    tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-small-generator")
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # Load pretrained Model
    disc_config = ElectraConfig.from_pretrained(f"google/electra-small-discriminator")
    gen_config = ElectraConfig.from_pretrained(f"google/electra-small-generator")

    generator = ElectraForMaskedLM(gen_config)
    discriminator = ElectraForPreTraining(disc_config)

    pretrained_model = Electra.ELECTRAModel(
        generator, discriminator, tokenizer.pad_token_id
    )
    pretrained_model.load_state_dict(torch.load("./pretrained/pytorch_model.bin"))
    model = Electra.ELECTRAClassificationModel(
        pretrained_model.discriminator, 0.1, num_labels, is_regression
    )

    # Compute Metrics
    def compute_metrics(p):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    data_collator = default_data_collator

    # TrainingArgs
    training_args = TrainingArguments(
        output_dir="./models",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        learning_rate=3e-4,
        logging_dir="./logs",
        report_to="wandb",
        remove_unused_columns=True,
        save_total_limit=2,
        num_train_epochs=3,  # 10 for RTE and STS, 2 for SQuAD, 3 for others
        load_best_model_at_end=False,
        weight_decay=0,
        save_steps=1000,
        adam_epsilon=1e-6,
        seed=SEED,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    train_result = trainer.train()
    metrics = train_result.metrics

    trainer.save_model()  # Saves the tokenizer too for easy upload

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    # Evaluation
    print("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    print("*** Test ***")

    metrics = trainer.evaluate(eval_dataset=test_dataset)

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)


def main():
    training_task = ["cola", "sst", "mrpc", "sts", "qqp", "mnli", "qnli", "rte"]
    for task in training_task:
        pretrain(task)


if __name__ == "__main__":
    main()
