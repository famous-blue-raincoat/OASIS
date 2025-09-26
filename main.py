
import random
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama import LlamaForCausalLM, LlamaConfig
import argparse
import torch
from dataset.evaluator import PPLMetric, compute_influence_loss
import os
from pathlib import Path
from dataset.utils import get_examples, get_loaders_
import torch.nn as nn 
import torch.nn.functional as F
from llama.modeling_llama import Linear_up

os.environ["TOKENIZERS_PARALLELISM"] = "false"


parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

# argument for parsing
parser.add_argument('--base_model', type=str, default="meta-llama/Llama-3.1-8B", help='base model name')
parser.add_argument('--sparsity', type=float, default=0.5, help='pruning ratio')
parser.add_argument("--save_model", nargs="?", const="../model", default=None, type=Path, metavar="DIR", help="Path to save the pruned model")


# argument for generation
parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
parser.add_argument('--top_p', type=float, default=0.95, help='top p')
parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')


# Calibration data
parser.add_argument('--data', type=str, default='c4')
parser.add_argument('--select_method', type=str, default='plain')
# parser.add_argument('--data_idx', type=int, default=-1)
parser.add_argument(
    '--data_idx', 
    type=int, 
    nargs='+',  # This is the key change! It means "accept one or more arguments"
    help='A list of data sample indices, separated by spaces'
    )
parser.add_argument('--num_examples', type=int, default=10)
parser.add_argument('--seq_len', type=int, default=128, help='calibration sequence length')
parser.add_argument('--verbose', action='store_true', help='verbose')
parser.add_argument('--select', action='store_true', help='select')
parser.add_argument('--get_kl', action='store_true', help='get kl divergence')



# argument for layer-wise pruning/column-wise pruning
parser.add_argument('--structure_prune', action='store_true', help='whether structure prune')
parser.add_argument('--global_pruning', action='store_true', help='get kl divergence')


parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=0)
parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

# Pruner
parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
parser.add_argument('--prune_type', type=str, default='NIRVANA', help='choose from [vectorize, param_second, param_first, param_mix]')


# general argument
parser.add_argument('--device', type=str, default="cuda", help='device')
parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
parser.add_argument('--test_before_prune', action='store_true', help='whether test after train')
parser.add_argument('--test_after_prune', action='store_true', help='whether test after prune')
parser.add_argument('--prune', action='store_true', help='whether prune')
parser.add_argument('--gamma', type=float, default=1.0, help='Attn vs MLP scaling factor')
parser.add_argument('--seed', type=int, default=42, help='seed')



args = parser.parse_args()

torch_version = float('.'.join(torch.__version__.split('.')[:2]))
args.torch_version = torch_version



def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def structure_prune_weight(linear_layer, neuron_indices, feature='row'):
      # Calculate the L1-norm of each neuron's weights        
      weight_mask = torch.ones_like(linear_layer.data, dtype=torch.bool)

      if feature == 'row':
        weight_mask[neuron_indices] = False
        
        linear_layer = linear_layer[weight_mask].clone().detach().view(
            linear_layer.size(0) - neuron_indices.numel(), linear_layer.size(1)
        )
      elif feature == 'column':
        weight_mask[:, neuron_indices] = False
        linear_layer = linear_layer[weight_mask].clone().detach().view(
          linear_layer.size(0), -1
        )
        
      
      del neuron_indices, weight_mask
      torch.cuda.empty_cache()
        
      
      return linear_layer
  
  
def get_pruning_idx(score, prune_type, n_pruned=-1, threshold=-1, global_pruning=False, head_dim:int=0, attn_type=None, kv_groups:int=0):

  if global_pruning:
    if threshold < 0:
      raise ValueError("Threshold must be set for global pruning")
    pruning_indices = torch.where(score<=threshold)[0]
  else:
    if n_pruned < 0:
      raise ValueError("Number of pruned neurons must be set for local pruning")
    _, pruning_indices = torch.topk(score, k=n_pruned, largest=False)
    
  if len(pruning_indices) == 0:return pruning_indices
  
  if len(pruning_indices) == len(score):pruning_indices=pruning_indices[:-1]
  
  
  if prune_type == 'attn':
    pruning_indices = torch.cat(
              [torch.tensor([j+head_dim*i for j in range(head_dim)])
              for i in pruning_indices], 0)
    
    if attn_type == 'q_proj' or attn_type == 'o_proj':
      pruning_indices = torch.cat(
              [torch.tensor([j+kv_groups*i for j in range(kv_groups)])
              for i in pruning_indices], 0)
  
  return pruning_indices

def structure_prune_bias(linear_layer, neuron_indices, feature='row'):
      # Calculate the L1-norm of each neuron's weights
      # print('before', linear_layer.shape)
        
      weight_mask = torch.ones_like(linear_layer.data, dtype=torch.bool)

      # print(neuron_indices, weight_mask.shape, linear_layer.size)

      weight_mask[neuron_indices] = False
      
      # print(linear_layer[weight_mask])
      
      linear_layer = linear_layer[weight_mask].clone().detach().view(
          linear_layer.size(0) - neuron_indices.numel()
      )
    
        
      # print(score.device, neuron_indices.device, weight_mask.device, linear_layer.device)
      del neuron_indices, weight_mask
      torch.cuda.empty_cache()
        
      # print('after', linear_layer.shape)
      
      return linear_layer



def main(args):
  

  tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
  example_prompts = get_loaders_(
                                name=args.data,
                                nsamples=args.num_examples,
                                seqlen=args.seq_len,
                                tokenizer=tokenizer,
                                idx=args.data_idx,
                                seed=args.seed,
                                select_method=args.select_method).to(args.device)
  
  # print(example_prompts)
  
  if example_prompts is None:
    print("No examples found")
    return
  
  u = nn.Parameter(torch.ones(args.num_examples, device='cuda') / args.num_examples)

  
  
  
  
  

  if 'llama' in args.base_model.lower():
    
    model_config = LlamaConfig.from_pretrained(args.base_model)
    model = LlamaForCausalLM.from_pretrained(
      args.base_model,
      config=model_config,
      low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
      device_map='auto',
      torch_dtype=torch.bfloat16,
    )
  elif 'qwen2' in args.base_model.lower():
    model_config = Qwen2Config.from_pretrained(args.base_model)
    model = Qwen2ForCausalLM.from_pretrained(
      args.base_model,
      config=model_config,
      low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
      device_map='auto',
      torch_dtype=torch.bfloat16,
      # attn_implementation="eager"
    )
  
  else:
    # model_config = AutoConfig.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(
      args.base_model,
      # config=model_config,
      low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
      device_map='auto',
      torch_dtype=torch.bfloat16,
    )
  
  # print(model)
  
  model.eval()
  
  before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  
  if args.test_before_prune:
    # ppl = PPLMetric(model, tokenizer, ['wikitext'], args.max_seq_len, device=args.eval_device)
    ppl = PPLMetric(model, tokenizer, ['wikitext', 'ptb','lambada'], args.max_seq_len, device=args.eval_device)
    # print("PPL before pruning: {}".format(ppl))
    print("PPL after pruning: {}, AVG {:.4f}".format(ppl, sum(ppl.values()) / len(ppl)))
  
  if args.prune:
    
    print("Start pruning with {} and {}".format(args.prune_type, args.data))    
    print("Start Backwarding in iterative steps = {}...".format(1))
  
    for epoch in range(1):
      if args.prune_type in ['NIRVANA', 'Pruner']:
        # out = model(example_prompts, labels=example_prompts)
        # loss = out[0]
        # loss.backward()
        cal_losses = []
        for i in range(args.num_examples):
            out = model(example_prompts[i].unsqueeze(0), labels=example_prompts[i].unsqueeze(0))
            loss = out[0]

            cal_losses.append(loss)

        cal_losses = torch.stack(cal_losses)
        L_cal = torch.sum(F.softmax(u, dim=0) * cal_losses)
        
        
        L_cal = torch.sum(u * cal_losses)
        L_cal.backward()
        
      
      num_heads = model.config.num_key_value_heads
      head_dim = model.config.hidden_size // model.config.num_attention_heads
      kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads
      
      score_norms = {}
      
      if args.structure_prune:      
      
        for m, p in list(model.named_parameters()):   
          if 'self_attn' not in m and 'mlp' not in m:
            continue
          
          if args.prune_type in ['NIRVANA']:
              score = torch.clone(p.grad * p.data).detach().abs_()
          elif args.prune_type == 'magnitude':
              score = torch.clone(p.data).detach().abs_()
          elif args.prune_type == 'Pruner':
              score = torch.clone(p.data * p.grad * p.data).detach()
              
          
          param_name = '.'.join(m.split('.')[:-2])
          
          # print(m, 'score', score, '\nparam', p.data, '\nparam grad', p.grad)
          
          if '.weight' in m:
            if m.split('.')[-2] in ['q_proj', 'k_proj', 'v_proj', 'gate_proj', 'up_proj']:
              score = torch.norm(score, p=1, dim=1)
                        
            elif m.split('.')[-2] in ['o_proj', 'down_proj']:
              score = torch.norm(score, p=1, dim=0)
              
          if 'self_attn' in m:
            # if '.weight' in m:
              
              
              score = score.view(num_heads, head_dim, -1)
              head_score = score.sum(dim=(1, 2))
              score_norms[param_name] = score_norms.get(param_name,0) + head_score
            # elif '.bias' in m:
            #   score = score.view(num_heads, head_dim, -1)
            #   head_score = score.sum(dim=(1, 2))
            #   score_norms[param_name] = score_norms.get(param_name,0) + head_score
                
          elif 'mlp' in m:
            if '.weight' in m:
              score_norms[param_name] = score_norms.get(param_name,0) + score 
                  
        
        # print(score_norms)
        
        if args.global_pruning:          
          
          attn_score = torch.cat([torch.flatten(v).to('cpu') for n,v in score_norms.items() if 'self_attn' in n])   
          mlp_score = torch.cat([torch.flatten(v).to('cpu') for n,v in score_norms.items() if 'mlp' in n])
          
          
          gamma = 1.0/args.gamma
          
          orginal_pruned = (head_dim * 2 * (kv_groups + 1) * num_heads + model.config.intermediate_size * 3) * args.sparsity  
          attn_mlp_ratio = (head_dim * 2 * (kv_groups + 1) * num_heads)/(model.config.intermediate_size * 3) * gamma * 3 / (head_dim * 2 * (kv_groups+1))  
          now_pruend = (head_dim * 2 * (kv_groups +1 ) + 1/attn_mlp_ratio * 3)   
          prune_attn = orginal_pruned/now_pruend*len(model.model.layers)
          prune_mlp = int(prune_attn/attn_mlp_ratio)
          prune_attn = int(prune_attn)
          
          
          if prune_attn>0:
            topk_imp_attn, _ = torch.topk(attn_score, k=prune_attn, largest=False)
          topk_imp_mlp, _ = torch.topk(mlp_score, k=prune_mlp, largest=False)
          
          attn_threshold, mlp_threshold = topk_imp_attn[-1] if prune_attn>0 else 0, topk_imp_mlp[-1]
          
          # print("Attn threshold:", attn_threshold, " MLP threshold:", mlp_threshold)
          
          for m, p in list(model.named_parameters()):      
            layer_name, param_name = m.rsplit('.', 1)
            layer = dict(model.named_modules())[layer_name]
            
            if 'self_attn' in m:
              
              attn_type = m.split('.')[-2]
              
              head_score = score_norms['.'.join(m.split('.')[:-2])]
              if '.weight' in m:
                if attn_type in ['q_proj', 'k_proj', 'v_proj']:              
                  
                  
                  prune_idx = get_pruning_idx(score=head_score,
                                              threshold=attn_threshold, 
                                              global_pruning=True,
                                              prune_type='attn',
                                              attn_type=attn_type,
                                              head_dim=head_dim,
                                              kv_groups=kv_groups)
                  
                  if len(prune_idx) == 0:continue
                  
                  pruned_param = structure_prune_weight(p, 
                                                        prune_idx,
                                                        'row')
                  
                  
                  
                  delattr(layer, param_name)
                  layer.out_features = pruned_param.size(0)
                  
                  layer.register_parameter(param_name, torch.nn.Parameter(pruned_param)) 
                  
                elif attn_type in ['o_proj']:
                  
                  prune_idx = get_pruning_idx(score=head_score,
                                              threshold=attn_threshold, 
                                              global_pruning=True,
                                              prune_type='attn',
                                              attn_type=attn_type,
                                              head_dim=head_dim,
                                              kv_groups=kv_groups)
                  
                  if len(prune_idx) == 0:continue
                  
                  pruned_param = structure_prune_weight(p, 
                                                        prune_idx,
                                                        'column')
                  
                  delattr(layer, param_name)
                  layer.in_features = pruned_param.size(1)
                  
                  layer.register_parameter(param_name, torch.nn.Parameter(pruned_param)) 
              elif '.bias' in m:
                
                # print(head_score)
                if attn_type in ['q_proj', 'k_proj', 'v_proj']:              
                  
                  # print(name)
                  
                  prune_idx = get_pruning_idx(score=head_score,
                                              # n_pruned=n_pruned, 
                                              threshold=attn_threshold, 
                                              global_pruning=True,
                                              prune_type='attn',
                                              attn_type=attn_type,
                                              head_dim=head_dim,
                                              kv_groups=kv_groups,)
                  
                  # print(prune_idx)
                  if len(prune_idx) == 0:continue
                  
                  
                  pruned_param = structure_prune_bias(p, prune_idx)
                  
                  
                  
                  delattr(layer, param_name)
                  layer.out_features = pruned_param.size(0)
                  
                  layer.register_parameter(param_name, torch.nn.Parameter(pruned_param))
              
            elif 'mlp' in m:
              if '.weight' in m:
                if m.split('.')[-3] == 'mlp':
                      prune_idx = get_pruning_idx(score=score_norms['.'.join(m.split('.')[:-2])],
                                                    threshold=mlp_threshold, 
                                                    global_pruning=True,
                                                    prune_type='mlp'
                                                    )
                      
                      if m.split('.')[-2] in ['gate_proj', 'up_proj']:
                        
                        pruned_param = structure_prune_weight(p, 
                                                              feature='row',
                                                              neuron_indices=prune_idx)
                        
                        delattr(layer, param_name)
                        layer.out_features = pruned_param.size(0)
                  
                      elif m.split('.')[-2] in ['down_proj']:
                        
                        pruned_param = structure_prune_weight(p, 
                                                              feature='column',
                                                              neuron_indices=prune_idx)
                        
                        delattr(layer, param_name)
                        layer.in_features = pruned_param.size(1)
                        
                      layer.register_parameter(param_name, torch.nn.Parameter(pruned_param))
              
        else:
          for m, p in list(model.named_parameters()):         
            
            layer_name, param_name = m.rsplit('.', 1)
            layer = dict(model.named_modules())[layer_name]
            
              
            # print(m)
            if 'self_attn' in m or 'mlp' in m:
              
              if m.split('.')[-3] == 'mlp':
                
                score = score_norms['.'.join(m.split('.')[:-2])]
                
                # n_pruned = len(score) - int(
                #     len(score) * args.sparsity
                # )
                
                n_pruned = int(len(score) * args.sparsity)
                
                prune_idx = get_pruning_idx(score=score,
                                            n_pruned=n_pruned,
                                            prune_type='mlp')
            
                if m.split('.')[-3] == 'mlp':
                    if m.split('.')[-2] in ['gate_proj', 'up_proj']:                      
                
                      pruned_param = structure_prune_weight(p, 
                                                            prune_idx,
                                                            'row',
                                                            )
                      
                      
                      delattr(layer, param_name)
                      layer.out_features = pruned_param.size(0)
                      
                      layer.register_parameter(param_name, torch.nn.Parameter(pruned_param)) 
                
                    elif m.split('.')[-2] in ['down_proj']:
                      
                
                      pruned_param = structure_prune_weight(p, 
                                                            prune_idx,
                                                            'column'
                                                            )
                      
                      
                      delattr(layer, param_name)
                      layer.in_features = pruned_param.size(1)
                      
                      layer.register_parameter(param_name, torch.nn.Parameter(pruned_param)) 
                  
              elif 'self_attn' in m:
                
                attn_type = m.split('.')[-2]
                
                head_score = score_norms['.'.join(m.split('.')[:-2])]
                
                # n_pruned = len(head_score) - int(
                #     len(head_score) * args.sparsity)
                
                n_pruned = int(len(head_score) * args.sparsity)
                
                prune_idx = get_pruning_idx(score=head_score,
                                              n_pruned=n_pruned,
                                              prune_type='attn',
                                              attn_type=attn_type,
                                              head_dim=head_dim,
                                              kv_groups=kv_groups)
                
                # print(head_score, prune_idx, n_pruned)
                if attn_type in ['q_proj', 'k_proj', 'v_proj']:
                  if '.weight' in m:
                    
                    pruned_param = structure_prune_weight(p, 
                                                          prune_idx,
                                                          'row')
                    
                    
                    
                  elif '.bias' in m:                   
                    
                    pruned_param = structure_prune_bias(p, prune_idx)
                    
                    
                    
                  delattr(layer, param_name)
                  layer.out_features = pruned_param.size(0)
                  
                  layer.register_parameter(param_name, torch.nn.Parameter(pruned_param))
                  
                elif attn_type in ['o_proj']:
                  pruned_param = structure_prune_weight(p, 
                                                        prune_idx,
                                                        'column',
                                                        )
                  
                  delattr(layer, param_name)
                  layer.in_features = pruned_param.size(1)
                  
                  layer.register_parameter(param_name, torch.nn.Parameter(pruned_param))             
                
          del score_norms, score
          # gc.collect()
          torch.cuda.empty_cache()
      
          after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)      

            

  if args.structure_prune:
    # print("Structure pruning finished")
    after_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("#Param before: {:,}, #Param after: {:,}, Ratio = {:.4f}%".format(before_pruning_parameters, after_pruning_parameters,  100.0*after_pruning_parameters/before_pruning_parameters))
    
  else:
    current_zeros=0
    total=0
    for n,p in model.named_parameters():
      if 'self_attn' in n or 'mlp' in n:
      # if any(i in n for i in name_to_skip):
      #   continue
        current_zeros += (p == 0).sum().item()
        total+=p.numel()
      
    print('Final Sparsity', current_zeros/total*100, '%')  
    
  
  # print(model)
  
  for idx, layer in enumerate(model.model.layers):    
      layer.self_attn.num_attention_heads = layer.self_attn.q_proj.weight.data.shape[0] // layer.self_attn.head_dim
  
  if args.test_after_prune:
    ppl = PPLMetric(model, tokenizer, ['wikitext', 'ptb','lambada'], args.max_seq_len, device=args.eval_device)
    print("PPL after pruning: {}, AVG {:.4f}".format(ppl, sum(ppl.values()) / len(ppl)))
      
  if args.save_model:
    for idx, layer in enumerate(model.model.layers): 
      model.config.modified_intermediate_dimension.append(layer.mlp.gate_proj.weight.shape[0])
      model.config.modified_head_num.append(layer.self_attn.num_attention_heads)
      
    model.config.pruned=True
    model.config.pruned_attn=True
    
    if args.select_method!='select':
      save_dir = args.save_model / f"{args.base_model}-{args.prune_type.split('_')[-1]}-{args.data}-{args.sparsity}-{args.select_method}-seed{args.seed}"
    else:
      save_dir = args.save_model / f"{args.base_model}-{args.prune_type.split('_')[-1]}-{args.data}-{args.sparsity}-{args.select_method}-seed{args.seed}"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    print("Model saved to", save_dir)




 
set_all_seeds(42)    
main(args)