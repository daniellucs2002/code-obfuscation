import pandas as pd
pd.set_option('max_colwidth',300)

from datasets import Dataset
from pathlib import Path
from transformers import logging as transformers_logging
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling

import torch
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--cuda', type=int, default=0, help='CUDA ID (0-3)')
parser.add_argument('--files', type=int, default=14, help='file numbers to be processed (1-14)')
parser.add_argument('--peek', type=int, default=0, help='print out samples during developing phase')
parser.add_argument('--versions', type=int, default=3, help='number of obfuscation versions in the comparison dataset')
parser.add_argument('--batch', type=int, default=32, help='batch size for the client side model')
parser.add_argument('--masked', type=float, default=0.1, help='probability to introduce <mask> token')
parser.add_argument('--topk', type=int, default=5, help='use top_k sampling method')

args = parser.parse_args()

device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

# step 1: prepare the raw data (python)

files = []
for i in range(0, args.files):
    files.extend(Path('../data/python/').glob(f'**/python_train_{i}.jsonl.gz'))

codes = pd.concat([pd.read_json(f,
                                orient='records',
                                compression='gzip',
                                lines=True)[['code_tokens']]
                   for f in files], sort=False)
codes['filtered_code_tokens'] = [[token for token in row if len(token) > 0 and token[0] != '#']
                                for row in codes['code_tokens']]
codes['code_string'] = [' '.join(row) for row in codes['filtered_code_tokens']]

if args.peek != 0:
    codes = codes.iloc[:args.peek]  # constrain to the first few batches
hf_codes = Dataset.from_pandas(codes)

transformers_logging.set_verbosity_error()
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
transformers_logging.set_verbosity_warning()
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
    remove_columns=['code_tokens', 'filtered_code_tokens']
)  # 'code_string'(original), 'input_ids', 'attention_mask'

# step 2: introduce <mask> token for comparison dataset

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=args.masked
)
