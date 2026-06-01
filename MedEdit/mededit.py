# import hydra
import os
import re
import sklearn
from easyeditor import BaseEditor
from easyeditor import  FTHyperParams, ROMEHyperParams, MEMITHyperParams, MENDTrainingHparams, MENDHyperParams,  LoRAHyperParams,  PMETHyperParams, MedLaSAHyperParams

from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaTokenizer
import numpy as np
from easyeditor import CounterFactDataset
from easyeditor import EditTrainer
import argparse
from collections import defaultdict
import json
import math
import torch

def metrics_compute(path):
    if isinstance(path, str):
        metrics = json.load(open(path, 'r', encoding='utf-8'))
    else:
        metrics = path
    results = defaultdict(list)

    for metric in metrics:
        for k,v in metric['post'].items():
            if k=='rewrite_acc' or k=='rephrase_acc': 
                results[k].append(v[0])
            if k=='fluency':
                results['fluency'].append(v['ngram_entropy'])
            if k=='locality' or k == 'portability':
                for k2,v2 in v.items():
                    results[k2].append(v2[0])
    print('------------------------------')
    result_list = []
    edit_success = []
    local_success = []
    for k,v in results.items():
        v = sum(v)/len(v)
        v = v * 100
        print(f"{k} : {v}")
        if k=='rewrite_acc' or k =='rephrase_acc':
            edit_success.append(v)
        if 'locality' in k:
            local_success.append(v)            
        result_list.append(round(v, 2))
    avg = (sum(edit_success)/len(edit_success) + sum(local_success)/len(local_success)) / 2
    print(f"avg : {avg}")
    result_list.append(round(avg, 2))
    print(result_list)

def MedCF_load(args):
    edit_data = json.load(open('./../MedCF/test.json', 'r', encoding='utf-8'))
    if args.sampledata:
        edit_data = edit_data[:args.sampledata]
    prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
    target_new = [edit_data_['target_new'] for edit_data_ in edit_data]
    ground_truth = [edit_data_['ground_truth'] for edit_data_ in edit_data]
    locality_inputs = {
        'locality_target':{
            'prompt': [edit_data_['locality_target_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_target_ground_truth'] for edit_data_ in edit_data]
        },
        'locality_mapping':{
            'prompt': [edit_data_['locality_mapping_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_mapping_ground_truth'] for edit_data_ in edit_data]
        },
        'locality_struc':{
            'prompt': [edit_data_['locality_struc_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_struc_ground_truth'] for edit_data_ in edit_data]
        },
        'locality_tokenSem':{
            'prompt': [edit_data_['locality_tokenSem_prompt'] for edit_data_ in edit_data],
            'ground_truth': [edit_data_['locality_tokenSem_ground_truth'] for edit_data_ in edit_data]
        },
    }
    return {'prompts':prompts, 'rephrase_prompts':rephrase_prompts, 'target_new':target_new, 'ground_truth':ground_truth,'subject':subject, 'locality_inputs':locality_inputs, 'portability_inputs':None}

def MedFE_load(args):
    tokenizer = LlamaTokenizer.from_pretrained("./../../LLM_checkpoint/chatdoctor-llama")
    def clip(text):
        tokens = tokenizer.tokenize(text)
        if len(tokens) > args.max_length:
            tokens = tokens[:args.max_length]
            text = tokenizer.convert_tokens_to_string(tokens)
        return text
    
    edit_data = json.load(open('./../MedFE/test.json', 'r', encoding='utf-8'))
    if args.sampledata:
        edit_data = edit_data[:args.sampledata]
    prompts = [edit_data_['prompt'] for edit_data_ in edit_data]
    subject = [edit_data_['subject'] for edit_data_ in edit_data]
    subject_type = [edit_data_['subject_type'] for edit_data_ in edit_data]
    topic_type = [edit_data_['topic_type'] for edit_data_ in edit_data]
    rephrase_prompts = [edit_data_['rephrase_prompt'] for edit_data_ in edit_data]
    target_new = [clip(edit_data_['target_new']) for edit_data_ in edit_data]
    locality_inputs = {
        'locality_topic':{
            'prompt': [edit_data_['locality_topic_prompt'] for edit_data_ in edit_data],
            'ground_truth': [clip(edit_data_['locality_topic_ground_truth']) for edit_data_ in edit_data]
        },
        'locality_tokenSem':{
            'prompt': [edit_data_['locality_tokenSem_prompt'] for edit_data_ in edit_data],
            'ground_truth': [clip(edit_data_['locality_tokenSem_ground_truth']) for edit_data_ in edit_data]
        },
    }
    return {'prompts':prompts, 'rephrase_prompts':rephrase_prompts, 'target_new':target_new, 'ground_truth':None, 'subject':subject, 'locality_inputs':locality_inputs,'portability_inputs':None}



def Args():
    parser = argparse.ArgumentParser(description='Example script with argparse')
    # Add arguments
    parser.add_argument('--trainedit', type=str,default='edit',)
    parser.add_argument('--method', type=str,default='FT', )
    parser.add_argument('--device', type=int,default=0, )
    parser.add_argument('--sampledata', type=int,default=0,)
    parser.add_argument('--max_length', type=int,default=200,)
    parser.add_argument('--dataset', type=str,default='MedCF',) # MedCF MedFE
    parser.add_argument('--lora_ver', type=str,default=None,)
    parser.add_argument('--lr', type=float, default=None,)
    parser.add_argument('--num_steps', type=int, default=70,)
    parser.add_argument("--alpha_dynamic", type=bool, default=True)
    parser.add_argument("--rank_dynamic", type=bool, default=True)
    parser.add_argument('--alpha0', type=float, default=64.0,)
    parser.add_argument('--rank0', type=int, default=16,)
    parser.add_argument('--norm', type=str, default='minmax',) # meanstd minmax
    parser.add_argument('--target_modules', type=str, nargs='+',default=["q_proj", "v_proj","k_proj", "o_proj","up_proj", "down_proj","gate_proj"])
    parser.add_argument('--model_name', type=str, default='chatdoctor',) # meditron
    parser.add_argument('--layers', type=int, nargs='+', default=None,)
    parser.add_argument('--allocation_strategy', type=str, default='original', choices=['original', 'adaptive'])
    parser.add_argument('--topk_layers', type=int, default=0,)
    parser.add_argument('--score_weights', type=float, nargs=3, default=[0.6, 0.4, 0.0])
    parser.add_argument('--editability_samples', type=int, default=8,)
    parser.add_argument('--risk_samples', type=int, default=8,)
    parser.add_argument('--dump_allocation', type=bool, default=True)


    args = parser.parse_args()
    print(args)
    return args

def write_result(args, metrics):
    _ensure_metrics_dir()
    metrics_filename = f'{args.dataset}-{args.method}'
    if args.lora_ver:
        metrics_filename += f'-{args.lora_ver}'
    if args.lr:
        metrics_filename += f'-{args.lr}'
    if args.num_steps:
        metrics_filename += f'-{args.num_steps}'
    if args.method == 'MedLaSA':
        if args.alpha_dynamic: metrics_filename += f'-alpha_dynamic'
        if args.rank_dynamic: metrics_filename += f'-rank_dynamic'
        metrics_filename += f'-{args.alpha0}'
        metrics_filename += f'-{args.rank0}'
        metrics_filename += f'-{args.norm}'
        tm_str = '-'.join(args.target_modules)
        metrics_filename += f'-{tm_str}'
        metrics_filename += f'-{args.allocation_strategy}'
        if args.topk_layers:
            metrics_filename += f'-topk{args.topk_layers}'
    metrics_filename += '.json'
    print(f'--------------dump {metrics_filename}----------------')
    with open(f'metrics_results/{metrics_filename}', 'w',encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False) 

class read_casual_tracing():
    def __init__(self,args,) -> None:
        self.args = args
        self.trace_root = self._resolve_tracing_root()
        self.samples = self._load_samples()
        self.filepath = os.path.join(self.trace_root, 'cases')
        self.num_layers = self._infer_num_layers()

    def _resolve_tracing_root(self):
        candidates = [
            f'../casual_tracing_data/{self.args.dataset}_causal_tracing',
            f'../casual_tracing_data/{self.args.dataset}_casual_tracing',
            f'../causal_tracing_data/{self.args.dataset}_causal_tracing',
            f'../causal_tracing_data/{self.args.dataset}_casual_tracing',
        ]
        for candidate in candidates:
            if os.path.isdir(candidate):
                return candidate
        raise FileNotFoundError(f'Can not find tracing directory for {self.args.dataset}. Tried: {candidates}')

    def _load_samples(self):
        json_candidates = [
            f'../casual_tracing_data/{self.args.dataset}_casual_tracing.json',
            f'../casual_tracing_data/{self.args.dataset}_causal_tracing.json',
            f'../causal_tracing_data/{self.args.dataset}_casual_tracing.json',
            f'../causal_tracing_data/{self.args.dataset}_causal_tracing.json',
        ]
        for candidate in json_candidates:
            if os.path.isfile(candidate):
                return json.load(open(candidate, 'r', encoding='utf-8'))
        raise FileNotFoundError(f'Can not find tracing metadata json for {self.args.dataset}. Tried: {json_candidates}')

    def _infer_num_layers(self):
        if not self.samples:
            raise ValueError('Tracing samples are empty.')
        known_id = self.samples[0]['known_id']
        sample_path = os.path.join(self.filepath, f'knowledge_{known_id}_attn.npz')
        impact = np.load(sample_path)
        return int(impact['scores'].shape[1])

    def normalization(self, data):
        data = np.asarray(data, dtype=np.float32)
        if self.args.norm == 'meanstd':
            std = np.std(data)
            if std == 0:
                return np.ones_like(data)
            return (data - np.mean(data)) / std  + 1
        else:
            span = np.max(data) - np.min(data)
            if span == 0:
                return np.ones_like(data)
            return (data - np.min(data)) / span

    def compute_alpha_score(self, path):
        impact = np.load(path)
        subject_range = impact['subject_range']
        score = impact['scores'][:-1,:]
        score = np.mean(score[subject_range[0]:subject_range[1],:],axis=0,keepdims=False)
        score = self.normalization(score)
        return score

    def _module_scope(self, target):
        if target in ["q_proj", "v_proj","k_proj", "o_proj"]:
            return 'attn'
        if target in ["up_proj", "down_proj","gate_proj"]:
            return 'mlp'
        raise ValueError(f'Unsupported target module: {target}')

    def _pattern_key(self, layer_idx, target):
        if self._module_scope(target) == 'attn':
            return f'model.layers.{layer_idx}.self_attn.{target}'
        return f'model.layers.{layer_idx}.mlp.{target}'

    def _aggregate_layer_scores(self, attn_scores, mlp_scores):
        attn_count = sum(1 for target in self.args.target_modules if self._module_scope(target) == 'attn')
        mlp_count = sum(1 for target in self.args.target_modules if self._module_scope(target) == 'mlp')
        if attn_count == 0:
            return np.asarray(mlp_scores, dtype=np.float32)
        if mlp_count == 0:
            return np.asarray(attn_scores, dtype=np.float32)
        return (attn_count * np.asarray(attn_scores, dtype=np.float32) + mlp_count * np.asarray(mlp_scores, dtype=np.float32)) / (attn_count + mlp_count)

    def _build_alpha_pattern(self, attn_scores, mlp_scores):
        alpha_pattern = {}
        for i in range(self.num_layers):
            for target in self.args.target_modules:
                score = attn_scores[i] if self._module_scope(target) == 'attn' else mlp_scores[i]
                alpha_pattern[self._pattern_key(i, target)] = float(min(max(self.args.alpha0 // 2, score * self.args.alpha0), self.args.alpha0 * 2))
        if not self.args.alpha_dynamic:
            return {}
        return alpha_pattern

    def _build_rank_pattern(self, attn_scores, mlp_scores):
        rank_pattern = {}
        for i in range(self.num_layers):
            for target in self.args.target_modules:
                score = attn_scores[i] if self._module_scope(target) == 'attn' else mlp_scores[i]
                rank = math.ceil(score * self.args.rank0)
                rank_pattern[self._pattern_key(i, target)] = int(min(max(self.args.rank0 // 2, rank), self.args.rank0 * 2))
        return rank_pattern

    def combine_scores(self, trace_scores, edit_scores=None, risk_scores=None):
        edit_scores = {} if edit_scores is None else edit_scores
        risk_scores = {} if risk_scores is None else risk_scores
        weights = self.args.score_weights
        combined = {}
        for scope in ('attn', 'mlp'):
            trace = self.normalization(trace_scores.get(scope, np.ones(self.num_layers)))
            edit = self.normalization(edit_scores.get(scope, np.ones(self.num_layers)))
            risk = self.normalization(risk_scores.get(scope, np.zeros(self.num_layers)))
            combined[scope] = weights[0] * trace + weights[1] * edit - weights[2] * risk
        return combined

    def compute_prompt2alpha(self,):
        prompt2alpha = {}
        for sample in self.samples:
            known_id = sample['known_id']
            attn_score = self.compute_alpha_score(os.path.join(self.filepath, f'knowledge_{known_id}_attn.npz'))
            mlp_score = self.compute_alpha_score(os.path.join(self.filepath, f'knowledge_{known_id}_mlp.npz'))
            alpha_pattern = self._build_alpha_pattern(attn_score, mlp_score)
            if self.args.dataset=='MedFE':
                prompt2alpha["Please provide an explanation for the following fact: \n " + sample['prompt']] = alpha_pattern
            elif self.args.dataset=='MedCF':
                prompt2alpha[sample['prompt']] = alpha_pattern
        return prompt2alpha

    def compute_global_score(self, samples, filepath, attn_or_mlp):
        scores = []
        for sample in samples:
            known_id = sample['known_id']
            impact = np.load(f'{filepath}/knowledge_{known_id}_{attn_or_mlp}.npz')
            score = impact['scores'][:-1,:]
            score = np.mean(score,axis=0,keepdims=False)
            scores.append(score)
        scores = np.stack(scores,axis=0)
        scores = np.mean(scores,axis=0,keepdims=False)
        scores = self.normalization(scores)
        return scores

    def compute_rank_pattern(self,):
        attn_scores = self.compute_global_score(self.samples, self.filepath, 'attn')
        mlp_scores = self.compute_global_score(self.samples, self.filepath, 'mlp')
        return self._build_rank_pattern(attn_scores, mlp_scores)

    def compute_trace_scores(self):
        return {
            'attn': self.compute_global_score(self.samples, self.filepath, 'attn'),
            'mlp': self.compute_global_score(self.samples, self.filepath, 'mlp'),
        }

    def compute_prompt_trace_scores(self):
        scores = {}
        for sample in self.samples:
            known_id = sample['known_id']
            prompt_key = "Please provide an explanation for the following fact: \n " + sample['prompt'] if self.args.dataset == 'MedFE' else sample['prompt']
            scores[prompt_key] = {
                'attn': self.compute_alpha_score(os.path.join(self.filepath, f'knowledge_{known_id}_attn.npz')),
                'mlp': self.compute_alpha_score(os.path.join(self.filepath, f'knowledge_{known_id}_mlp.npz')),
            }
        return scores

    def select_topk_layers(self, layer_scores):
        if not self.args.topk_layers or self.args.topk_layers >= len(layer_scores):
            return list(range(len(layer_scores)))
        indices = np.argsort(layer_scores)[::-1][:self.args.topk_layers]
        return sorted(int(i) for i in indices)


def _ensure_metrics_dir():
    os.makedirs('metrics_results', exist_ok=True)


def dump_allocation_summary(args, summary):
    if not args.dump_allocation:
        return
    _ensure_metrics_dir()
    filename = f'{args.dataset}-{args.method}-allocation'
    if args.lora_ver:
        filename += f'-{args.lora_ver}'
    filename += f'-{args.allocation_strategy}.json'
    with open(f'metrics_results/{filename}', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)


def _set_target_module_grad_flags(model, target_modules):
    pattern = re.compile(r'model\.layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.weight$')
    for param in model.parameters():
        param.requires_grad_(False)
    for name, param in model.named_parameters():
        match = pattern.search(name)
        if match and match.group(3) in target_modules:
            param.requires_grad_(True)


def _build_supervised_batch(tok, prompts, targets, device):
    eos_token = tok.decode(tok.eos_token_id)
    full_prompt = [f"{p} {l} {eos_token}" for p, l in zip(prompts, targets)]
    prompt_ids = tok(list(prompts), return_tensors="pt", padding=True, truncation=True)["input_ids"]
    num_prompt_toks = [int((i != tok.pad_token_id).sum()) for i in prompt_ids]
    tokens = tok(full_prompt, return_tensors="pt", padding=True, truncation=True)
    tokens["labels"] = tokens["input_ids"].clone()
    num_pad_toks = [int((i == tok.pad_token_id).sum()) for i in tokens["labels"]]
    for i in range(len(prompts)):
        tokens["labels"][i][num_pad_toks[i]:num_pad_toks[i] + num_prompt_toks[i]] = -100
    tokens["labels"][tokens["input_ids"] == tok.pad_token_id] = -100
    return tokens.to(device)


def _collect_locality_examples(dataset, limit):
    prompts, targets = [], []
    locality_inputs = dataset.get('locality_inputs') or {}
    for locality_group in locality_inputs.values():
        prompts.extend(locality_group['prompt'])
        targets.extend(locality_group['ground_truth'])
        if len(prompts) >= limit:
            break
    return prompts[:limit], targets[:limit]


def _compute_gradient_scores(model_path, args, prompts, targets):
    if not prompts or not targets:
        return None
    device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
    model = model.to(device)
    model.train()
    _set_target_module_grad_flags(model, args.target_modules)

    attn_scores = defaultdict(float)
    mlp_scores = defaultdict(float)
    layer_pattern = re.compile(r'model\.layers\.(\d+)\.(self_attn|mlp)\.([^.]+)\.weight$')
    max_samples = min(len(prompts), len(targets))

    for prompt, target in zip(prompts[:max_samples], targets[:max_samples]):
        model.zero_grad(set_to_none=True)
        batch = _build_supervised_batch(tok, [prompt], [target], device)
        loss = model(**batch).loss
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            match = layer_pattern.search(name)
            if not match or match.group(3) not in args.target_modules:
                continue
            layer_idx = int(match.group(1))
            scope = match.group(2)
            grad_norm = float(param.grad.norm().item())
            if scope == 'self_attn':
                attn_scores[layer_idx] += grad_norm
            else:
                mlp_scores[layer_idx] += grad_norm

    num_layers = max(max(attn_scores.keys(), default=-1), max(mlp_scores.keys(), default=-1)) + 1
    if num_layers <= 0:
        return None
    attn = np.array([attn_scores[i] for i in range(num_layers)], dtype=np.float32)
    mlp = np.array([mlp_scores[i] for i in range(num_layers)], dtype=np.float32)
    del model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'attn': attn, 'mlp': mlp}


def _resize_scores(scores, num_layers):
    if scores is None:
        return None
    resized = {}
    for scope, values in scores.items():
        if len(values) == num_layers:
            resized[scope] = values
        elif len(values) > num_layers:
            resized[scope] = values[:num_layers]
        else:
            padded = np.zeros(num_layers, dtype=np.float32)
            padded[:len(values)] = values
            resized[scope] = padded
    return resized


def _compute_adaptive_allocation(args, hparams, dataset, ct):
    trace_scores = ct.compute_trace_scores()
    prompt_trace_scores = ct.compute_prompt_trace_scores()
    edit_scores = _compute_gradient_scores(
        hparams.model_name,
        args,
        dataset['prompts'][:args.editability_samples],
        dataset['target_new'][:args.editability_samples],
    )
    risk_prompts, risk_targets = _collect_locality_examples(dataset, args.risk_samples)
    risk_scores = _compute_gradient_scores(hparams.model_name, args, risk_prompts, risk_targets)
    edit_scores = _resize_scores(edit_scores, ct.num_layers)
    risk_scores = _resize_scores(risk_scores, ct.num_layers)

    combined_global = ct.combine_scores(trace_scores, edit_scores=edit_scores, risk_scores=risk_scores)
    combined_layer_scores = ct._aggregate_layer_scores(combined_global['attn'], combined_global['mlp'])
    selected_layers = ct.select_topk_layers(combined_layer_scores)

    rank_pattern = ct._build_rank_pattern(combined_global['attn'], combined_global['mlp']) if args.rank_dynamic else {}
    if not args.rank_dynamic:
        hparams.rank = args.rank0

    prompt2alpha = {}
    for prompt_key, prompt_trace in prompt_trace_scores.items():
        combined_prompt = ct.combine_scores(prompt_trace, edit_scores=edit_scores, risk_scores=risk_scores)
        prompt2alpha[prompt_key] = ct._build_alpha_pattern(combined_prompt['attn'], combined_prompt['mlp'])

    summary = {
        'strategy': args.allocation_strategy,
        'weights': {
            'trace': args.score_weights[0],
            'editability': args.score_weights[1],
            'risk': args.score_weights[2],
        },
        'selected_layers': selected_layers,
        'trace_scores': {k: np.asarray(v).tolist() for k, v in trace_scores.items()},
        'editability_scores': None if edit_scores is None else {k: np.asarray(v).tolist() for k, v in edit_scores.items()},
        'risk_scores': None if risk_scores is None else {k: np.asarray(v).tolist() for k, v in risk_scores.items()},
        'combined_scores': {k: np.asarray(v).tolist() for k, v in combined_global.items()},
        'layer_scores': np.asarray(combined_layer_scores).tolist(),
    }
    return prompt2alpha, rank_pattern, selected_layers, summary


def _extend_prompt2alpha_from_dataset(args, dataset, prompt2alpha):
    extended = dict(prompt2alpha)
    for idx, prompt in enumerate(dataset['prompts']):
        alpha_pattern = prompt2alpha.get(prompt)
        if alpha_pattern is None and args.dataset == 'MedFE':
            prefixed = "Please provide an explanation for the following fact: \n " + prompt
            alpha_pattern = prompt2alpha.get(prefixed)
            if alpha_pattern is not None:
                extended[prompt] = alpha_pattern
        if alpha_pattern is None:
            continue
        if idx < len(dataset['rephrase_prompts']):
            extended[dataset['rephrase_prompts'][idx]] = alpha_pattern
        locality_inputs = dataset.get('locality_inputs') or {}
        for locality_group in locality_inputs.values():
            if idx < len(locality_group['prompt']):
                extended[locality_group['prompt'][idx]] = alpha_pattern
    return extended


def edit_med_llm(args):
    if args.dataset == 'MedCF': dataset = MedCF_load(args)
    elif args.dataset == 'MedFE': dataset = MedFE_load(args)
    model = globals()[f'{args.method}HyperParams']

    hparams = model.from_hparams(f'./hparams/{args.method}-llama.yaml') 
    hparams.model_name = f"./../../LLM_checkpoint/{args.model_name}-llama"
    if hparams.model_parallel:
        hparams.device = 'cuda'
    if args.method=='MEND':
        hparams.archive = f'./results/{args.dataset}/models/MEND/{args.model_name}-llama'
        hparams.results_dir = f'./results/{args.dataset}'
        hparams.tokenizer_name = f'./../../LLM_checkpoint/{args.model_name}-llama'
    

    if args.lora_ver:
        hparams.lora_ver = args.lora_ver
    if args.lr:
        hparams.lr = args.lr

    if args.layers:
        hparams.layers = args.layers

    if args.method == 'MedLaSA':
        if args.lora_ver=='AdaLoRA':
            args.rank_dynamic = False
        ct = read_casual_tracing(args)

        hparams.target_modules = args.target_modules
        hparams.num_steps = args.num_steps
        allocation_summary = None
        if args.allocation_strategy == 'adaptive':
            prompt2alpha, hparams.rank_pattern, selected_layers, allocation_summary = _compute_adaptive_allocation(args, hparams, dataset, ct)
            hparams.layers = selected_layers
        else:
            if args.rank_dynamic:
                hparams.rank_pattern = ct.compute_rank_pattern()
            else:
                hparams.rank_pattern = {}
                hparams.rank = args.rank0
            prompt2alpha = ct.compute_prompt2alpha()
        if not args.alpha_dynamic:
            hparams.lora_alpha = args.alpha0
        prompt2alpha = _extend_prompt2alpha_from_dataset(args, dataset, prompt2alpha)
        if allocation_summary is not None:
            dump_allocation_summary(args, allocation_summary)

        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.mededit(
            prompts=dataset['prompts'],
            rephrase_prompts=dataset['rephrase_prompts'],
            target_new=dataset['target_new'],
            ground_truth=dataset['ground_truth'],
            subject=dataset['subject'],
            locality_inputs=dataset['locality_inputs'],
            portability_inputs=dataset['portability_inputs'],
            prompt2alpha=prompt2alpha,
            args=args,
            train_ds=None,
            keep_original_weight=True,
            test_generation=True,
        )
    else :
        editor = BaseEditor.from_hparams(hparams)
        metrics, edited_model, _ = editor.edit(
            prompts=dataset['prompts'],
            rephrase_prompts=dataset['rephrase_prompts'],
            target_new=dataset['target_new'],
            ground_truth=dataset['ground_truth'],
            subject=dataset['subject'],
            locality_inputs=dataset['locality_inputs'],
            portability_inputs=dataset['portability_inputs'],
            train_ds=None,
            keep_original_weight=True,
            test_generation=True,
        )
    metrics_compute(metrics)
    write_result(args, metrics)
    
    return metrics, edited_model


def MEND_Train_Llama(args):
    model = globals()[f'{args.method}TrainingHparams']
    training_hparams = model.from_hparams(f'./hparams/{args.method}-train-llama.yaml')
    training_hparams.results_dir = f'./results/{args.dataset}'
    train_ds = CounterFactDataset(f'./../MEND_training/{args.dataset}/train.json', config=training_hparams)
    eval_ds = CounterFactDataset(f'./../MEND_training/{args.dataset}/valid.json', config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()


if __name__=='__main__':
    args = Args()    
    if args.trainedit=='train':
        MEND_Train_Llama(args)
    else:
        edit_med_llm(args)
    # metrics_compute('./metrics_results/MedCF-FT.json')
