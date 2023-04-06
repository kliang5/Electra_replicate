from .train import tokenizer

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
    #result["labels"] = result["input_ids"].copy()
    return result