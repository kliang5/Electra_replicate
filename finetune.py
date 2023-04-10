import random
import datasets
import evaluate
import numpy as np
from datasets import load_dataset

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import tensor as T

import transformers
from transformers import (
    Trainer,
    ElectraConfig,
    ElectraTokenizerFast,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    TrainingArguments,
    default_data_collator,
)

SEED = 203
# Tokenizer
tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-small-generator")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def pretrain(task_name):
    # task_name = "qnli"
    sentence1_key, sentence2_key = task_to_keys[task_name]
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

    non_label_column_names = [
        name for name in raw_datasets["train"].column_names if name != "label"
    ]

    # Preprocessing the raw_datasets
    def preprocess_function(examples):
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)

        result["label"] = examples["label"]
        return result

    raw_datasets = raw_datasets.map(
        preprocess_function,
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

    ## ELECTRA Model, input: generator and discriminator
    ## Joinly trained both models
    class ELECTRAModel(nn.Module):
        def __init__(
            self, generator, discriminator, pad_token_id, dis_loss_weight=50.0
        ):
            super().__init__()
            self.generator, self.discriminator, self.pad_token_id = (
                generator,
                discriminator,
                pad_token_id,
            )
            self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0.0, device="cuda:0"), torch.tensor(1.0, device="cuda:0")
            )
            self.dis_loss_weight = dis_loss_weight
            self.gen_loss_fc = nn.CrossEntropyLoss()  # maybe Flatten
            self.dis_loss_fc = nn.BCEWithLogitsLoss()

        def forward(
            self, masked_ids, attention_mask, token_type_ids, is_masked, labels
        ):
            # masked_ids: masked input ids (Tensor[int]): (B, L)
            # attention_mask: attention_mask from data collator (Tensor[int]): (B, L)
            # token_type_ids: token_type_ids of the tokenizer
            # is_masked: whether the position is get masked (Tensor[boolean]): (B, L)
            # labels: (Tensor[int]): (batch_size, seq_length) -100 for the unmasked

            # Feed into generator
            gen_logits = self.generator(
                masked_ids, attention_mask, token_type_ids
            ).logits  # (B, L, vocab size)
            masked_gen_logits = gen_logits[is_masked, :]

            with torch.no_grad():
                # gumble arg-max to have tokenized predicted word
                pred_toks = self.gumbel_softmax(masked_gen_logits)
                # inputs for discriminator
                gen_ids = masked_ids.clone()
                gen_ids[is_masked] = pred_toks  # (B, L)
                # labels for discriminator
                is_replaced = is_masked.clone()
                is_replaced[is_masked] = pred_toks != labels[is_masked]  # (B, L)

            # Feed into discrminator
            dis_logits = self.discriminator(
                gen_ids, attention_mask, token_type_ids
            ).logits  # (B, L, vocab size)

            # Loss function of Electra
            gen_loss = self.gen_loss_fc(masked_gen_logits.float(), labels[is_masked])
            # Discriminator Loss
            dis_logits = dis_logits.masked_select(attention_mask == 1)
            is_replaced = is_replaced.masked_select(attention_mask == 1)
            disc_loss = self.dis_loss_fc(dis_logits.float(), is_replaced.float())
            loss = gen_loss + disc_loss * self.dis_loss_weight

            return {
                "loss": loss,
                "masked_gen_logits": masked_gen_logits,
                "gen_logits": gen_logits,
                "dis_logits": dis_logits,
            }

        def gumbel_softmax(self, logits):
            return (logits.float() + self.gumbel_dist.sample(logits.shape)).argmax(
                dim=-1
            )

    # Just the discriminator for fine-tuning
    class ELECTRAClassificationModel(nn.Module):
        def __init__(self, discriminator, classifier_dropout, num_labels):
            super().__init__()
            self.discriminator = discriminator
            self.num_labels = num_labels
            self.dropout = nn.Dropout(classifier_dropout)
            self.classifier = nn.Linear(128, num_labels)
            self.loss_fn = nn.CrossEntropyLoss() if not is_regression else nn.MSELoss()

        def forward(self, input_ids, attention_mask, token_type_ids, labels):
            dis_logits = self.discriminator(
                input_ids, attention_mask, token_type_ids
            ).logits  # (B, L, vocab size)

            dis_logits = self.dropout(dis_logits)
            logits = self.classifier(dis_logits)

            if is_regression:
                loss = self.loss_fn(logits.squeeze(), labels.squeeze())
            else:
                loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))

            return {
                "loss": loss,
                "logits": logits,
            }

    # Load pretrained Model
    disc_config = ElectraConfig.from_pretrained(f"google/electra-small-discriminator")
    gen_config = ElectraConfig.from_pretrained(f"google/electra-small-generator")

    generator = ElectraForMaskedLM(gen_config)
    discriminator = ElectraForPreTraining(disc_config)

    pretrained_model = ELECTRAModel(generator, discriminator, tokenizer.pad_token_id)
    pretrained_model.load_state_dict(torch.load("./pretrained/pytorch_model.bin"))
    model = ELECTRAClassificationModel(pretrained_model.discriminator, 0.1, num_labels)

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
