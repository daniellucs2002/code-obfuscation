# utils/file_operations.py

import json

class Example(object):
    """A single example for rating on the server side."""
    def __init__(self, idx, source, target, obfuscate):
        self.idx = idx
        self.source = source
        self.target = target
        self.obfuscate = obfuscate

def read_summarize_examples(filename, data_num):
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
