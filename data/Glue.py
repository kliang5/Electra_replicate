import numpy as np

import transformers
from transformers import (
    ElectraTokenizerFast,
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

# Preprocessing the raw_datasets
def preprocess_function(task_name, examples):
    sentence1_key, sentence2_key = task_to_keys[task_name]
    args = (
        (examples[sentence1_key],)
        if sentence2_key is None
        else (examples[sentence1_key], examples[sentence2_key])
    )
    result = tokenizer(*args, padding="max_length", max_length=128, truncation=True)

    result["label"] = examples["label"]
    return result
