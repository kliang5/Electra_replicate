import os
import wandb
import torch
from torch import nn
from datasets import load_dataset, load_from_disk
from transformers import (
    Trainer,
    ElectraConfig,
    ElectraTokenizerFast,
    ElectraForMaskedLM,
    ElectraForPreTraining,
    TrainingArguments,
)

from transformers import AdamW
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

SEED = 203

# Data
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])


sequence_len = 128


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # Drop the small remainder
    if total_length >= sequence_len:
        total_length = (total_length // sequence_len) * sequence_len
    # Split by chunks of sequence_len.
    result = {
        k: [t[i : i + sequence_len] for i in range(0, total_length, sequence_len)]
        for k, t in concatenated_examples.items()
    }
    # result["labels"] = result["input_ids"].copy()
    return result


tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-small-generator")

if os.path.isdir("/scratch/webtext_tokenized"):
    dataset = load_from_disk(dataset_path="/scratch/webtext_tokenized")
else:
    dataset = load_dataset("openwebtext", cache_dir="/scratch/webtext")
    dataset = dataset["train"].train_test_split(shuffle=True, test_size=0.1)["test"]

    dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        remove_columns=dataset.column_names,
    )
    print("FINISHED TOKENIZATION")
    dataset = dataset.map(group_texts, batched=True, num_proc=8)
    # Save the pre processed dataset
    dataset.save_to_disk(dataset_path="/scratch/webtext_tokenized")


# Electra DataCollator
@dataclass
class ElectraDataCollator:
    tokenizer: ElectraTokenizerFast
    mlm: bool = True
    mlm_probability: float = 0.15
    tf_experimental_compile: bool = False
    return_tensors: str = "pt"

    def __call__(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt")

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)

        batch["masked_ids"], batch["labels"] = self.torch_mask_tokens(
            batch["input_ids"], special_tokens_mask=special_tokens_mask
        )
        batch["is_masked"] = batch["labels"] != -100
        batch.pop("input_ids")
        return batch

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


""" ELECTRA Model, input: generator and discriminator
Joinly trained both models """


class ELECTRAModel(nn.Module):
    def __init__(self, generator, discriminator, pad_token_id, dis_loss_weight=50.0):
        super().__init__()
        self.generator, self.discriminator, self.pad_token_id = (
            generator,
            discriminator,
            pad_token_id,
        )
        self.dis_loss_weight = dis_loss_weight
        self.gen_loss_fc = nn.CrossEntropyLoss()
        self.dis_loss_fc = nn.BCEWithLogitsLoss()

    def forward(
        self,
        masked_ids,
        attention_mask,
        token_type_ids,
        is_masked,
        labels,
        compute_metrics=True,
    ):
        """input_ids: (Tensor[int]): (batch_size, seq_length)
        masked_ids: masked input ids (Tensor[int]): (B, L)
        attention_mask: attention_mask from data collator (Tensor[int]): (B, L)
        token_type_ids: token_type_ids of the tokenizer
        is_masked: whether the position is get masked (Tensor[boolean]): (B, L)
        """

        # Feed into generator
        gen_logits = self.generator(
            masked_ids, attention_mask, token_type_ids
        ).logits  # (B, L, vocab size)
        masked_gen_logits = gen_logits[is_masked, :]

        with torch.no_grad():
            # gumble arg-max to have tokenized predicted word
            pred_toks = masked_gen_logits.argmax(-1)
            # inputs for discriminator
            gen_ids = masked_ids.clone()
            gen_ids[is_masked] = pred_toks  # (B, L)
            # labels for discriminator
            is_replaced = is_masked.clone()
            is_replaced[is_masked] = pred_toks != labels[is_masked]  # (B, L)

        # Feed into discrminator
        disc_logits = self.discriminator(
            gen_ids, attention_mask, token_type_ids
        ).logits  # (B, L, vocab size)

        # Loss function of Electra
        gen_loss = self.gen_loss_fc(masked_gen_logits.float(), labels[is_masked])
        # Discriminator Loss
        # disc_logits = disc_logits.masked_select(attention_mask == 1)
        # is_replaced = is_replaced.masked_select(attention_mask == 1)
        disc_loss = self.dis_loss_fc(disc_logits.float(), is_replaced.float())
        loss = gen_loss + disc_loss * self.dis_loss_weight

        if compute_metrics:
            # gen_correct = pred_toks == labels[is_masked]
            disc_correct = (disc_logits.sigmoid() >= 0) == is_replaced
            # gen_acc = torch.mean(torch.sum(gen_correct, dim=1) / disc_logits.shape[1])
            disc_acc = torch.mean(torch.sum(disc_correct, dim=1) / disc_logits.shape[1])
            wandb.log(
                {
                    "disc_loss": disc_loss,
                    "disc_acc": disc_acc,
                    "gen_loss": gen_loss,
                    #    "gen_acc": gen_acc,
                }
            )

        return {
            "loss": loss,
            "masked_gen_logits": masked_gen_logits,
            "gen_logits": gen_logits,
            "disc_logits": disc_logits,
        }


# Train
disc_config = ElectraConfig.from_pretrained(f"google/electra-small-discriminator")
gen_config = ElectraConfig.from_pretrained(f"google/electra-small-generator")

generator = ElectraForMaskedLM(gen_config)
discriminator = ElectraForPreTraining(disc_config)
discriminator.electra.embeddings = generator.electra.embeddings
generator.generator_lm_head.weight = generator.electra.embeddings.word_embeddings.weight
pad_token_id = tokenizer.pad_token_id
# Data Collator
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
data_collator = ElectraDataCollator(tokenizer=tokenizer, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir="./models",
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=128,
    per_device_eval_batch_size=128,
    learning_rate=5e-2,  # changed from paper
    lr_scheduler_type="cosine_with_restarts",  # changed from paper "linear"
    logging_dir="./logs",
    report_to="wandb",
    remove_unused_columns=False,
    save_total_limit=2,
    load_best_model_at_end=False,
    warmup_steps=625,  # changed from paper
    weight_decay=0.01,
    save_steps=1000,
    adam_epsilon=1e-6,
    adam_beta1=0.9,
    adam_beta2=0.999,
    max_steps=62500,
    seed=SEED,
)

model = ELECTRAModel(generator, discriminator, pad_token_id)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./pretrained")
