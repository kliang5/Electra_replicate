import os
import wandb
import torch
from torch import nn
import argparse
from datasets import load_dataset, load_from_disk
from transformers import (
    Trainer,
    ElectraConfig,
    ElectraTokenizerFast,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    TrainingArguments,
)
from model import Electra
from data import OWT

SEED = 203


def preprocess():
    if os.path.isdir("/scratch/webtext_tokenized"):
        dataset = load_from_disk(dataset_path="/scratch/webtext_tokenized")
    else:
        dataset = load_dataset("openwebtext", cache_dir="/scratch/webtext")
        dataset = dataset["train"].train_test_split(shuffle=True, test_size=0.1)["test"]

        dataset = dataset.map(
            OWT.preprocess_function,
            batched=True,
            num_proc=8,
            remove_columns=dataset.column_names,
        )
        print("FINISHED TOKENIZATION")
        dataset = dataset.map(OWT.group_texts, batched=True, num_proc=8)
        # Save the pre processed dataset
        dataset.save_to_disk(dataset_path="/scratch/webtext_tokenized")

    return dataset


def train(args, dataset):
    # Train
    # Load the tokenizer
    tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-small-generator")

    # Load the config &model
    disc_config = ElectraConfig.from_pretrained(f"google/electra-small-discriminator")
    gen_config = ElectraConfig.from_pretrained(f"google/electra-small-generator")
    generator = ElectraForMaskedLM(gen_config)
    discriminator = ElectraForPreTraining(disc_config)

    # Set shared embeddings
    discriminator.electra.embeddings = generator.electra.embeddings
    generator.generator_lm_head.weight = (
        generator.electra.embeddings.word_embeddings.weight
    )
    pad_token_id = tokenizer.pad_token_id
    # Data Collator
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    data_collator = OWT.ElectraDataCollator(tokenizer=tokenizer, mlm_probability=0.15)

    training_args = TrainingArguments(
        output_dir="./models",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,  # changed from paper
        lr_scheduler_type="cosine_with_restarts",  # changed from paper "linear"
        logging_dir="./logs",
        report_to="wandb",
        remove_unused_columns=False,
        save_total_limit=2,
        load_best_model_at_end=False,
        warmup_steps=args.warmup_steps,  # changed from paper
        weight_decay=args.weight_decay,
        save_steps=1000,
        adam_epsilon=1e-6,
        adam_beta1=0.9,
        adam_beta2=0.999,
        max_steps=args.max_steps,
        seed=SEED,
    )

    model = Electra.ELECTRAModel(generator, discriminator, pad_token_id)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model("./pretrained")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int, help="batch size")
    parser.add_argument("--lr", default=5e-2, type=float, help="learning rate")
    parser.add_argument("--warmup_steps", default=625, type=int, help="warmup_steps")
    parser.add_argument("--weight_decay", default=0.01, type=float, help="weight_decay")
    parser.add_argument(
        "--max_steps", default=62500, type=int, help="max training steps"
    )
    args = parser.parse_args()

    dataset = preprocess()
    train(args, dataset)
