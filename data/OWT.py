import torch
from transformers import (
    ElectraTokenizerFast,
)

from transformers import AdamW
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

SEED = 203
sequence_len = 128
tokenizer = ElectraTokenizerFast.from_pretrained(f"google/electra-small-generator")

# Data
def preprocess_function(examples):
    return tokenizer([" ".join(x) for x in examples["text"]])


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
