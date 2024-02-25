# config/bleu_config.py

class BleuConfig(object):
    data_num = -1
    model_type = 'codet5'
    beam_size = 10
    task = 'summarize'
    max_target_length = 128
    res_dir = '/home/ugproj/daniel/CodeSearchNet/proj/dataset'
    lang = 'python'
    split_tag = 'test'
    criteria = 'best_bleu'
