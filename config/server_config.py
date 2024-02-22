# config/server_config.py

class ServerConfig(object):
    model_type = 'codet5'
    config_name = ''
    model_name_or_path = 'Salesforce/codet5-base'
    tokenizer_name = 'Salesforce/codet5-base'
    load_model_path = None
    finetune_path = '/home/ugproj/daniel/CodeSearchNet/CodeT5/CodeT5/finetuned_models/summarize_python_codet5_base.bin'
