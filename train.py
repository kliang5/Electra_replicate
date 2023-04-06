from transformers import DataCollatorForLanguageModeling, Trainer, ElectraConfig, ElectraTokenizerFast, 
ElectraForMaskedLM, ElectraForPreTraining, TrainingArguments
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch import tensor as T
from datasets import load_dataset

# Load dataset 
dataset = load_dataset("openwebtext", cache_dir='/content/drive/MyDrive/ELECTRA/Electra/datasets')
tokenizer = ElectraTokenizerFast.from_pretrained(f'google/electra-small-generator')

# 
tokenizer.pad_token = tokenizer.eos_token
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)