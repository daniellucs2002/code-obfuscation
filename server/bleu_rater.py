# server side, calculating bleu score

import sys
sys.path.append('/home/ugproj/daniel/CodeSearchNet/proj')

import time
import multiprocessing
from logger import logger
from tqdm import tqdm
import torch
import os
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

from utils.file_operations import *
from utils.model_operations import *

from config.server_config import *
from config.bleu_config import *

from evaluator import smooth_bleu
from evaluator.bleu import _bleu
from evaluator.CodeBLEU import calc_code_bleu

class BleuRater(object):
    def __init__(self):

        self.t0 = time.time()

        # load examples from file for rating
        self.client_output = '/home/ugproj/daniel/CodeSearchNet/proj/dataset/client_output.jsonl'
        self.examples = read_summarize_examples(self.client_output)
        calc_stats(self.examples)

        # load server side CodeT5 model
        server_config = ServerConfig()
        _, self.model, self.tokenizer = build_or_load_gen_model(server_config)
        self.device = 'cuda:2'
        self.model.to(self.device)

        self.eval_batch_size = 24
        self.max_source_length = 256

        logger.info("  " + "***** Rating *****")
        logger.info("  Batch size = %d", self.eval_batch_size)

    # prepare examples for rating (eval_examples, eval_data)
    def prepare_examples(self):
        tuple_examples = [(example, idx, self.tokenizer, self.max_source_length) 
                          for idx, example in enumerate(self.examples)]
        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            features = pool.map(
                convert_examples_to_features,
                tqdm(tuple_examples, total=len(tuple_examples))
            )  # from tokens to ids
        # features(dev): 320 examples * 2 code snippets * 256 token_ids
        all_source_ids = torch.tensor(features, dtype=torch.long)
        self.data = TensorDataset(all_source_ids)
        # torch.Size([320, 2, 256])
        logger.info(f"  features shape: {self.data.tensors[0].shape}")

    # calculate bleu score
    def eval_bleu_epoch(self, args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
        logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
        logger.info("  Num examples = %d", len(eval_examples))
        logger.info("  Batch size = %d", self.eval_batch_size)
        eval_sampler = SequentialSampler(eval_data)
        if args.data_num == -1:
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size,
                                        num_workers=4, pin_memory=True)
        else:
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.eval_batch_size)

        model.eval()
        pred_ids = []
        bleu, codebleu = 0.0, 0.0
        for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
            source_ids = batch[0].to(self.device)
            source_mask = source_ids.ne(tokenizer.pad_token_id)
            with torch.no_grad():
                if args.model_type == 'roberta':
                    preds = model(source_ids=source_ids, source_mask=source_mask)

                    top_preds = [pred[0].cpu().numpy() for pred in preds]
                else:
                    preds = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        use_cache=True,
                                        num_beams=args.beam_size,
                                        early_stopping=args.task == 'summarize',
                                        max_length=args.max_target_length)
                    top_preds = list(preds.cpu().numpy())
                pred_ids.extend(top_preds)

        pred_nls = [tokenizer.decode(id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for id in pred_ids]

        output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
        gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
        src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

        if args.task in ['defect']:
            target_dict = {0: 'false', 1: 'true'}
            golds = [target_dict[ex.target] for ex in eval_examples]
            eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
            result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

            with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
                for pred_nl, gold in zip(pred_nls, eval_examples):
                    f.write(pred_nl.strip() + '\n')
                    f1.write(target_dict[gold.target] + '\n')
                    f2.write(gold.source.strip() + '\n')
                logger.info("Save the predictions into %s", output_fn)
        else:
            dev_accs, predictions = [], []
            with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
                for pred_nl, gold in zip(pred_nls, eval_examples):
                    dev_accs.append(pred_nl.strip() == gold.target.strip())
                    if args.task in ['summarize']:
                        # for smooth-bleu4 evaluation
                        predictions.append(str(gold.idx) + '\t' + pred_nl)
                        f.write(str(gold.idx) + '\t' + pred_nl.strip() + '\n')
                        f1.write(str(gold.idx) + '\t' + gold.target.strip() + '\n')
                        f2.write(str(gold.idx) + '\t' + gold.source.strip() + '\n')
                    else:
                        f.write(pred_nl.strip() + '\n')
                        f1.write(gold.target.strip() + '\n')
                        f2.write(gold.source.strip() + '\n')

            if args.task == 'summarize':
                (goldMap, predictionMap) = smooth_bleu.computeMaps(predictions, gold_fn)
                bleu = round(smooth_bleu.bleuFromMaps(goldMap, predictionMap)[0], 2)
            else:
                bleu = round(_bleu(gold_fn, output_fn), 2)
                if args.task in ['concode', 'translate', 'refine']:
                    codebleu = calc_code_bleu.get_codebleu(gold_fn, output_fn, args.lang)

            result = {'em': np.mean(dev_accs) * 100, 'bleu': bleu}
            if args.task == 'concode':
                result['codebleu'] = codebleu * 100

        logger.info("***** Eval results *****")
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(round(result[key], 4)))

        return result

if __name__ == "__main__":
    bleu_rater = BleuRater()
    bleu_rater.prepare_examples()  # before feeding into the model
    bleu_config = BleuConfig()
    bleu_rater.eval_bleu_epoch(bleu_config, TensorDataset(bleu_rater.data[:, 0, :][0]), bleu_rater.examples, bleu_rater.model, bleu_rater.tokenizer, bleu_config.split_tag, bleu_config.criteria)
