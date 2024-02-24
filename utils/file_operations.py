# utils/file_operations.py

import json
import numpy as np
from logger import logger

class Example(object):
    """A single example for rating on the server side."""
    def __init__(self, idx, source, target, obfuscate):
        self.idx = idx
        self.source = source
        self.target = target
        self.obfuscate = obfuscate

def read_summarize_examples(filename, data_num=0):
    """Read examples from filename. (up to data_num)"""
    examples = []
    with open(filename, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            js = json.loads(line)
            if 'idx' not in js:
                js['idx'] = idx
            code = js['code_string'].replace('\n', ' ')
            code = ' '.join(code.strip().split())  # original code snippet
            nl = js['target_summarize'].replace('\n', ' ')
            nl = ' '.join(nl.strip().split())  # target python code summarization
            
            obfuscate = [
                ' '.join(s.replace('\n', ' ').strip().split()) 
                for s in js['obfuscated_versions']
            ]  # obfuscated code snippets

            examples.append(
                Example(
                    idx=idx,
                    source=code,
                    target=nl,
                    obfuscate=obfuscate
                )
            )

            if idx + 1 == data_num:
                break
            
    return examples

def calc_stats(examples, tokenizer=None, is_tokenize=False):
    avg_src_len = []
    avg_trg_len = []
    avg_src_len_tokenize = []
    avg_trg_len_tokenize = []
    for ex in examples:
        if is_tokenize:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
            avg_src_len_tokenize.append(len(tokenizer.tokenize(ex.source)))
            avg_trg_len_tokenize.append(len(tokenizer.tokenize(str(ex.target))))
        else:
            avg_src_len.append(len(ex.source.split()))
            avg_trg_len.append(len(str(ex.target).split()))
    if is_tokenize:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))
        logger.info("[TOKENIZE] avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    np.mean(avg_src_len_tokenize), np.mean(avg_trg_len_tokenize), max(avg_src_len_tokenize),
                    max(avg_trg_len_tokenize))
    else:
        logger.info("Read %d examples, avg src len: %d, avg trg len: %d, max src len: %d, max trg len: %d",
                    len(examples), np.mean(avg_src_len), np.mean(avg_trg_len), max(avg_src_len), max(avg_trg_len))

def convert_examples_to_features(item):
    example, _, tokenizer, max_source_length = item
    source_str = example.source  # single source
    source_list = example.obfuscate  # a list of obfuscated sources
    source_list = [source_str] + source_list

    # encode the source and obfuscated sources using the tokenizer
    source_ids_list = tokenizer.batch_encode_plus(
        source_list,
        max_length = max_source_length,
        padding = 'max_length',
        truncation = True
    )

    for src_id in source_ids_list['input_ids']:
        assert src_id.count(tokenizer.eos_token_id) == 1

    return source_ids_list['input_ids']
