import os
os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

os.environ['CUDA_VISIBLE_DEVICES'] = '3,0,2'

from tqdm import tqdm
import pandas as pd
tqdm.pandas()

from utils.edit_distance import word_level_edit_distance
from server.bleu_rater import *
from multiprocessing import Pool

from transformers import AutoTokenizer
from datasets import Dataset
from pathlib import Path

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

import wandb
import torch

# configure wandb for better logging
wandb.init(project="project")

batch_sz = 16

config = PPOConfig(
    # client side model: PolyCoder with 160M, 0.4B, or 2.7B parameters
    model_name="NinedayWang/PolyCoder-0.4B",

    learning_rate=1.41e-5,
    log_with="wandb",
    mini_batch_size=batch_sz,
    batch_size=batch_sz,
)

def build_dataset(config):

    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    files = []
    files.append(Path('/data/common/CodeSearchNet/code2nl/python/test.jsonl'))  # test

    codes = pd.concat([pd.read_json(f, orient='records', lines=True)[['code_tokens', 'docstring_tokens']]
                    for f in files], sort=False)
    codes['filtered_code_tokens'] = [[token for token in row if len(token) > 0 and token[0] != '#']
                                    for row in codes['code_tokens']]
    codes['query'] = [' '.join(row) for row in codes['filtered_code_tokens']]
    codes['label'] = [' '.join(row) for row in codes['docstring_tokens']]
    ds = Dataset.from_pandas(codes)

    def tokenize_fn(example):
        return tokenizer(
            example['query'],
            truncation=True,
        )

    ds = ds.map(
        tokenize_fn,
        batched=True,
        num_proc=4,
        remove_columns=['code_tokens', 'filtered_code_tokens', 'docstring_tokens']
    )  # 'query'(input to client), 'input_ids', 'attention_mask'; 'label'
    ds.set_format(type="torch")

    # filter out examples that are too long
    ds = ds.filter(lambda example: example['input_ids'].shape[0] < 200)

    return ds


dataset = build_dataset(config)  # num_rows: 11669

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)

tokenizer.pad_token = tokenizer.eos_token

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)

# device = ppo_trainer.accelerator.device
# if ppo_trainer.accelerator.num_processes == 1:
#     device = 1 if torch.cuda.is_available() else "cpu"
device = 3 if torch.cuda.is_available() else "cpu"

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
}  # generation settings

# setup RM at server side
bleu_config = BleuConfig()
bleu_rater = BleuRater(config.batch_size)

# the training loop

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
    query_tensors = batch["input_ids"]

    #### Get response from the PolyCoder
    response_tensors = []
    for query in query_tensors:
        generation_kwargs["max_new_tokens"] = query.shape[0]
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-generation_kwargs["max_new_tokens"]:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    #### Get rewards for the current batch generation
    paired_list = list(zip(batch["query"], batch["response"]))  # diff between query and response
    with Pool() as pool:
        distances = pool.map(word_level_edit_distance, paired_list)

    bleu_rater.prepare_examples(batch["response"], batch["label"])  # before feeding into the model
    # then, get bleu score for each example
    codebleu, bleulist = bleu_rater.eval_bleu_epoch(bleu_config, TensorDataset(bleu_rater.data[:, 0, :][0]), bleu_rater.examples, 
                                            bleu_rater.model, bleu_rater.tokenizer, bleu_config.split_tag, bleu_config.criteria)

    bleu_rater.prepare_examples(batch["query"], batch["label"])  # before feeding into the model
    codebleu_ref, bleulist_ref = bleu_rater.eval_bleu_epoch(bleu_config, TensorDataset(bleu_rater.data[:, 0, :][0]), bleu_rater.examples, 
                                            bleu_rater.model, bleu_rater.tokenizer, bleu_config.split_tag, bleu_config.criteria)

    rewards = [torch.tensor(2 * d + b - bref) for (d, b, bref) in zip(distances, bleulist, bleulist_ref)]

    #### Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    ppo_trainer.log_stats(stats, batch, rewards)

    print(f"code bleu: {codebleu}, bleu ref: {codebleu_ref} ({codebleu-codebleu_ref}), diff: {statistics.mean(distances)}, rewards: {torch.mean(torch.stack(rewards))}")


# save the model
access_token = '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'
torch.save(model, "/home/ugproj/daniel/CodeSearchNet/proj/experiment00.pt")
