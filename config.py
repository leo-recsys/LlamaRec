
import argparse
import torch

RAW_DATASET_ROOT_FOLDER = 'data'
EXPERIMENT_ROOT = 'experiments'
STATE_DICT_KEY = 'model_state_dict'
OPTIMIZER_STATE_DICT_KEY = 'optimizer_state_dict'
PROJECT_NAME = 'llmrec'


def set_template(args):
    if args.dataset_code == None:
        print('******************** Dataset Selection ********************')
        dataset_code = {'1': 'ml-100k', 'b': 'beauty', 'g': 'games'}
        args.dataset_code = dataset_code[input('Input 1 for ml-100k, b for beauty and g for games: ')]

    if args.dataset_code == 'ml-100k':
        args.bert_max_len = 200
    else:
        args.bert_max_len = 50

    if 'llm' in args.model_code:
        batch = 16 if args.dataset_code == 'ml-100k' else 12
        args.lora_micro_batch_size = batch
    else:
        batch = 16 if args.dataset_code == 'ml-100k' else 64

    args.train_batch_size = batch
    args.val_batch_size = batch
    args.test_batch_size = batch

    if torch.cuda.is_available(): args.device = 'cuda'
    else: args.device = 'cpu'
    args.optimizer = 'AdamW'
    args.lr = 0.001
    args.weight_decay = 0.01
    args.enable_lr_schedule = False
    args.decay_step = 10000
    args.gamma = 1.
    args.enable_lr_warmup = False
    args.warmup_steps = 100

    args.metric_ks = [1, 5, 10, 20, 50]
    args.rerank_metric_ks = [1, 5, 10]
    args.best_metric = 'Recall@10'
    args.rerank_best_metric = 'NDCG@10'

    args.bert_num_blocks = 2
    args.bert_num_heads = 2
    args.bert_head_size = None

args_dict = {
    'dataset_code': None,
    'min_rating': 0,
    'min_uc': 5,
    'min_sc': 5,
    'seed': 42,
    'train_batch_size': 64,
    'val_batch_size': 64,
    'test_batch_size': 64,
    'num_workers': 8,
    'sliding_window_size': 1.0,
    'negative_sample_size': 10,
    'device': 'cuda',
    'num_epochs': 500,
    'optimizer': 'AdamW',
    'weight_decay': None,
    'adam_epsilon': 1e-09,
    'momentum': None,
    'lr': 0.001,
    'max_grad_norm': 5.0,
    'enable_lr_schedule': True,
    'decay_step': 10000,
    'gamma': 1,
    'enable_lr_warmup': True,
    'warmup_steps': 100,
    'val_strategy': 'iteration',
    'val_iterations': 500,
    'early_stopping': True,
    'early_stopping_patience': 20,
    'metric_ks': [ 1, 5, 10, 20, 50 ],
    'rerank_metric_ks': [ 1, 5, 10 ],
    'best_metric': 'Recall@10',
    'rerank_best_metric': 'NDCG@10',
    'use_wandb': False,
    'model_code': None,
    'bert_max_len': 50,
    'bert_hidden_units': 64,
    'bert_num_blocks': 2,
    'bert_num_heads': 2,
    'bert_head_size': 32,
    'bert_dropout': 0.2,
    'bert_attn_dropout': 0.2,
    'bert_mask_prob': 0.25,
    'llm_base_model': 'meta-llama/Llama-2-7b-hf',
    'llm_base_tokenizer': 'meta-llama/Llama-2-7b-hf',
    'llm_max_title_len': 32,
    'llm_max_text_len': 1536,
    'llm_max_history': 20,
    'llm_train_on_inputs': False,
    'llm_negative_sample_size': 19,
    'llm_system_template': 'Given user history in chronological order, recommend an item from the candidate pool with its index letter.',
    'llm_input_template': 'User history: {}; \n Candidate pool: {}',
    'llm_load_in_4bit': True,
    'llm_retrieved_path': None,
    'llm_cache_dir': None,
    'lora_r': 8,
    'lora_alpha': 32,
    'lora_dropout': 0.05,
    'lora_target_modules': [ 'q_proj', 'v_proj' ],
    'lora_num_epochs': 1,
    'lora_val_iterations': 100,
    'lora_early_stopping_patience': 20,
    'lora_lr': 0.0001,
    'lora_micro_batch_size': 16
}

args = argparse.Namespace(**args_dict)
print("args:", args)
