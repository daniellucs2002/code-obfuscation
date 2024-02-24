# utils/model_operations.py

import torch
import numpy as np
from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                          BartConfig, BartForConditionalGeneration, BartTokenizer,
                          T5Config, T5ForConditionalGeneration, T5Tokenizer)
from logger import logger

MODEL_CLASSES = {'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
                 't5': (T5Config, T5ForConditionalGeneration, T5Tokenizer),
                 'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
                 'bart': (BartConfig, BartForConditionalGeneration, BartTokenizer)}

def get_model_size(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_size = sum([np.prod(p.size()) for p in model_parameters])
    return "{}M".format(round(model_size / 1e+6))

def build_or_load_gen_model(server_config):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[server_config.model_type]
    config = config_class.from_pretrained(server_config.config_name if server_config.config_name else server_config.model_name_or_path)
    tokenizer = tokenizer_class.from_pretrained(server_config.tokenizer_name)
    if server_config.model_type == 'roberta':
        # encoder = model_class.from_pretrained(args.model_name_or_path, config=config)
        # decoder_layer = nn.TransformerDecoderLayer(d_model=config.hidden_size, nhead=config.num_attention_heads)
        # decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        # model = Seq2Seq(encoder=encoder, decoder=decoder, config=config,
        #                 beam_size=args.beam_size, max_length=args.max_target_length,
        #                 sos_id=tokenizer.cls_token_id, eos_id=tokenizer.sep_token_id)
        assert False
    else:
        model = model_class.from_pretrained(server_config.model_name_or_path)

    logger.info("Finish loading model [%s] from %s", get_model_size(model), server_config.model_name_or_path)

    if server_config.load_model_path is not None:
        logger.info("Reload model from {}".format(server_config.load_model_path))
        model.load_state_dict(torch.load(server_config.load_model_path))
    
    # finetuned model should be reloaded here
    logger.info("Reload model from {}".format(server_config.finetune_path))
    model.load_state_dict(torch.load(server_config.finetune_path))

    return config, model, tokenizer
