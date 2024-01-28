import pandas as pd
pd.set_option('max_colwidth',300)

from datasets import Dataset
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling

import torch
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# step 1: prepare the raw data

files = sorted(Path('../data/python/').glob('**/*.jsonl'))
codes = pd.concat([pd.read_json(f,
                                orient='records',
                                lines=True)[['code_tokens']]
                   for f in files], sort=False)
codes['filtered_code_tokens'] = [[token for token in row if len(token) > 0 and token[0] != '#']
                                for row in codes['code_tokens']]
codes['code_string'] = [' '.join(row) for row in codes['filtered_code_tokens']]

codes = codes.iloc[:64]  # constrain to the first two batches
hf_codes = Dataset.from_pandas(codes)

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
model.to(device)

def tokenize_fn(example):
    return tokenizer(
        example['code_string'],
        truncation=True,
        padding="max_length"
    )

hf_codes = hf_codes.map(
    tokenize_fn,
    batched=True,
    num_proc=4,
    remove_columns=hf_codes.column_names
)

# step 2: introduce <mask> token

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=0.1
)

dataloader = DataLoader(
    hf_codes,
    batch_size=32,
    collate_fn=data_collator
)

print(tokenizer.mask_token_id)  # <mask> token
print(hf_codes[0]['input_ids'])  # before processing
print(next(iter(dataloader))['input_ids'][0])  # after processing

# step 3: feed masked snippet into the model

for batch in dataloader:
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        output = model(**batch)
    print(output.logits.shape)  # torch.Size([32, 512, 50265])
    input_ids = batch['input_ids']
    predictions = torch.argmax(output.logits, dim=-1)

    # iterate over each sample in the batch
    for i in range(input_ids.size(0)):
        masked_pos = (input_ids[i] == tokenizer.mask_token_id).nonzero(as_tuple=False)
        for pos in masked_pos:
            input_ids[i, pos] = predictions[i, pos]

print(next(iter(dataloader))['input_ids'][0])  # final output of the client model
print(dataloader.dataset)
