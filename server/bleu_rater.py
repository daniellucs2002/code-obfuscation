# server side, calculating bleu score

import sys
sys.path.append('/home/ugproj/daniel/CodeSearchNet/proj')

import time
import multiprocessing
from logger import logger
from tqdm import tqdm

from utils.file_operations import *
from utils.model_operations import *

from config.server_config import *

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

        self.eval_batch_size = 48
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

if __name__ == "__main__":
    bleu_rater = BleuRater()
    bleu_rater.prepare_examples()  # before feeding into the model
