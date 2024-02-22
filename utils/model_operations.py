# utils/model_operations.py

import torch
import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
import logging

logger = logging.getLogger(__name__)

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def build_or_load_gen_model(config):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[config.model_type]
    config = config_class.from_pretrained(config.config_name if config.config_name else config.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(config.tokenizer_name)
    if config.model_type == 'roberta':
        # encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        # decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        # model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
        #                 beam_size=args.beam_size, max_length=args.max_target_length,
        #                 sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        assert False
    else:
        model = model_class.from_pretrained(config.model_name_or_path)

    logger.info("Finish loading model [%s] from %s", get_model_size(model), config.model_name_or_path)

    if config.load_model_path is not None:
        logger.info("Reload model from {}".format(config.load_model_path))
        model.load_state_dict(torch.load(config.load_model_path))
    
    # finetuned model should be reloaded here
    logger.info("Reload model from {}".format(config.finetune_path))
    model.load_state_dict(torch.load(config.finetune_path))

    return config, model, tokenizer
