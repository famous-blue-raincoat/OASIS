import os
os.environ["PYTHONHASHSEED"] = "42" 
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" 
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import random
import numpy as np
from transformers import AutoTokenizer
from llama import LlamaForCausalLM, LlamaConfig

import argparse
from tqdm import tqdm
import wandb


import torch
torch.use_deterministic_algorithms(True)
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from dataset.evaluator import PPLMetric
import os
from pathlib import Path
from dataset.utils import get_examples, get_loaders_
import torch.nn as nn 
import torch.nn.functional as F

# from llama.modeling_llama import Linear_up
from qwen2.modeling_qwen2 import Linear_up



def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

set_all_seeds(42)    

parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')

# argument for parsing
parser.add_argument('--base_model', type=str, default="meta-llama/Llama-3.1-8B", help='base model name')
parser.add_argument('--sparsity', type=float, default=0.5, help='pruning ratio')
parser.add_argument("--save_model", nargs="?", const="../model", default=None, type=Path, metavar="DIR", help="Path to save the pruned model")


# argument for generation
# parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
parser.add_argument('--top_p', type=float, default=0.95, help='top p')
parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')


# Calibration data
parser.add_argument('--data', type=str, default='c4')
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
parser.add_argument('--get_kl', action='store_true', help='get kl divergence')


parser.add_argument('--select', action='store_true', help='select')
parser.add_argument('--train', action='store_true', help='whether train u')
parser.add_argument('--train_samples', type=int, default=128, help='numer of data to train u')
parser.add_argument('--select_type', type=str, default='global', help='choose from [global, local]')


parser.add_argument('--lr', type=float, default=1e-3, help='learning rate for u')



# argument for layer-wise pruning/column-wise pruning
parser.add_argument('--structure_prune', action='store_true', help='whether structure prune')

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
parser.add_argument('--test_before_prune', action='store_true', help='whether test before prune')
parser.add_argument('--test_after_prune', action='store_true', help='whether test after prune')
parser.add_argument('--prune', action='store_true', help='whether prune')
parser.add_argument('--gamma', type=float, default=1.0, help='Attn vs MLP scaling factor')
parser.add_argument('--seed', type=int, default=42, help='seed')

# Train
parser.add_argument('--temperature', type=float, default=0.8, help='temperature')
parser.add_argument('--noise_level', type=float, default=0.1, help='noise_level')

args = parser.parse_args()

torch_version = float('.'.join(torch.__version__.split('.')[:2]))
args.torch_version = torch_version

# torch.autograd.set_detect_anomaly(True)



args = parser.parse_args()

torch_version = float('.'.join(torch.__version__.split('.')[:2]))
args.torch_version = torch_version

# torch.autograd.set_detect_anomaly(True)

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



class AlphaScheduler:
    """
    A simple scheduler to linearly anneal the alpha value for the soft mask.
    """
    def __init__(self, initial_alpha, final_alpha, total_steps):
        """
        Args:
            initial_alpha (float): The starting value of alpha.
            final_alpha (float): The final value of alpha.
            total_steps (int): The total number of training steps over which to anneal.
        """
        self.initial_alpha = initial_alpha
        self.final_alpha = final_alpha
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        """
        Update the internal step counter. Should be called once per training step.
        """
        if self.current_step < self.total_steps:
            self.current_step += 1

    def get_alpha(self):
        """
        Calculate and return the current alpha value based on the current step.
        """
        # 线性插值计算alpha
        progress = self.current_step / self.total_steps
        current_alpha = self.initial_alpha + (self.final_alpha - self.initial_alpha) * progress
        return min(current_alpha, self.final_alpha) # 确保不会超过 final_alpha

def compute_structured_mask_dict(model, param_grads_dict, sparsity=0.5, alpha=10.0, pruning_type='neuron', method='NIRVANA'):
  """
  Calculate structured pruning masks based on aggregated importance scores.
  
  Args:
    model (nn.Module): Your model.
    param_grads_dict (dict): Mapping from parameter names to gradients.
    sparsity (float): Pruning ratio.
    alpha (float): Sharpness of the sigmoid function for soft pruning.
    pruning_type (str): Currently supports 'neuron'.

  Returns:
    dict: Mapping from module names to structured masks.
  """
  mask_dict, score_dict = {}, {}
  
  num_heads = model.config.num_key_value_heads
  head_dim = model.config.hidden_size // model.config.num_attention_heads
  kv_groups = model.config.num_attention_heads // model.config.num_key_value_heads
  
  # print(param_grads_dict)

  for module_name_in_model, module in model.named_modules():
    if isinstance(module, Linear_up):  # Assuming the pruning target is the Linear_up layer
      # print(module_name_in_model, param_grads_dict)
      
      weight_param_name = f"{module_name_in_model}.weight"
      bias_param_name = f"{module_name_in_model}.bias"
      
      # print(module_name_in_model, bias_param_name)

      # print("START", module_name_in_model, score_dict)
      if weight_param_name in param_grads_dict or bias_param_name in param_grads_dict:
        W = module.weight
        G = param_grads_dict[weight_param_name]
        
        if W is None or G is None:
          continue

        if method == 'NIRVANA':
          saliency = (W * G).abs_()  # shape: [out_features, in_features]
        elif method == 'Pruner':
          saliency = (W * G * W)
          
        # print(module_name_in_model, saliency)s
        
        # print(weight_param_name, 'score', saliency, '\nparam', W, '\nparam grad', G)

        if module_name_in_model.split('.')[-1] in ['gate_proj', 'up_proj', 'q_proj', 'k_proj', 'v_proj']:
          neuron_saliency = torch.norm(saliency, p=1, dim=1)  # shape: [out_features]
          
        elif module_name_in_model.split('.')[-1] in ['down_proj', 'o_proj']:
          neuron_saliency = torch.norm(saliency, p=1, dim=0)
        
        else:
          raise ValueError(f"Unsupported module type for pruning: {module_name_in_model.split('.')[-1]}")
        
        param_name = '.'.join(module_name_in_model.split('.')[:-1])
        if 'self_attn' in module_name_in_model:
          # Aggregate the scores for each head.
          neuron_saliency = neuron_saliency.view(num_heads, head_dim, -1)
          head_score = neuron_saliency.sum(dim=(1, 2))
          score_dict[param_name] = score_dict.get(param_name,0) + head_score
        elif 'mlp' in module_name_in_model:
          # For MLP layers, we can directly use the neuron saliency.
          score_dict[param_name] = score_dict.get(param_name, 0) + neuron_saliency
        # print("ADD WEIGHT", module_name_in_model, score_dict)
        
        if bias_param_name in param_grads_dict:
          # print("BIAS")
          W = module.bias
          G = param_grads_dict[bias_param_name]
          
          if W is None or G is None:
            continue

          if method == 'NIRVANA':
            saliency = (W * G).abs_()  # shape: [out_features, in_features]
          elif method == 'Pruner':
            saliency = (W * G * W)
            
          
            
          else:
            raise ValueError(f"Unsupported module type for pruning: {module_name_in_model.split('.')[-1]}")
          
          # print(bias_param_name, 'score', saliency, '\nparam', W, '\nparam grad', G)
          
          param_name = '.'.join(module_name_in_model.split('.')[:-1])
          neuron_saliency = neuron_saliency.view(num_heads, head_dim, -1)
          head_score = neuron_saliency.sum(dim=(1, 2))
          score_dict[param_name] = score_dict.get(param_name,0) + head_score
        
        
        # print("ADD BIAS", module_name_in_model, score_dict)

      
  
  # print(score_dict)
  
  if method == 'NIRVANA':
    
    attn_score = torch.cat([torch.flatten(v).to('cpu') for n,v in score_dict.items() if 'self_attn' in n])   
    mlp_score = torch.cat([torch.flatten(v).to('cpu') for n,v in score_dict.items() if 'mlp' in n])
    
    
    
    
    # Gamma for scaling between attention and MLP
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
    # attn_threshold, mlp_threshold = topk_imp_attn[-1].detach().clone() if prune_attn>0 else 0, topk_imp_mlp[-1].detach().clone()
    attn_threshold, mlp_threshold = topk_imp_attn[-1] if prune_attn>0 else 0, topk_imp_mlp[-1]
    
    # print("Attn threshold:", attn_threshold, " MLP threshold:", mlp_threshold)

    
    for module_name_in_model, module in model.named_modules():
      if isinstance(module, Linear_up): 
          # print(module_name_in_model)
          param_name = '.'.join(module_name_in_model.split('.')[:-1])
          neuron_saliency = score_dict.get(param_name, None)
          
          
          
          # Create soft mask based on the threshold, neuron_saliency still connects to the computation graph
          # This way the gradient of L_target can flow through mask -> neuron_saliency -> saliency -> G -> u
          if 'self_attn' in module_name_in_model:
            soft_mask_1d = torch.sigmoid((alpha/50) * (neuron_saliency - attn_threshold))
          elif 'mlp' in module_name_in_model:
            soft_mask_1d = torch.sigmoid(alpha * (neuron_saliency - mlp_threshold))
          
          # print(soft_mask_1d.shape)
          
          if 'self_attn' in module_name_in_model:
            soft_mask_1d = torch.cat(
                      [torch.tensor([i for j in range(head_dim)])
                      for i in soft_mask_1d], 0)
            
            if module_name_in_model.split('.')[-1] in ['q_proj', 'o_proj']:
              soft_mask_1d = torch.cat(
                      [torch.tensor([i for j in range(kv_groups)])
                      for i in soft_mask_1d], 0)
              
            # print(soft_mask_1d.shape)
          # if '.weight' in module_name_in_model:
          
          weight_mask = None
          bias_mask = None
              
          if module_name_in_model.split('.')[-1] in ['gate_proj', 'up_proj', 'q_proj', 'k_proj', 'v_proj']:
            weight_mask = soft_mask_1d.unsqueeze(1)  # a.k.a. row-wise mask
            # if module_name_in_model.split('.')[-1] in ['gate_proj', 'up_proj']:continue
            if module_name_in_model.split('.')[-1] in ['q_proj', 'k_proj', 'v_proj']:bias_mask = soft_mask_1d
            # bias_mask = soft_mask_1d
            
          elif module_name_in_model.split('.')[-1] in ['down_proj', 'o_proj']:
            weight_mask = soft_mask_1d.unsqueeze(0)  # a.k.a. column-wise mask
            

          # print(f"Module: {module_name_in_model}, Mask shape: {structured_mask.shape}, Sparsity: {1 - structured_mask.mean().item()}, Smallest value: {structured_mask.min().item()}, Average value: {structured_mask.mean().item()}, Largest value: {structured_mask.max().item()}")
          # mask_dict[module.name] = structured_mask
          
          
          
          mask_dict[module.name] = {'weight': weight_mask, 'bias': bias_mask}


  elif method == 'Pruner':
    # print(score_dict)
    for module_name_in_model, module in model.named_modules():
      if isinstance(module, Linear_up): 
          # print(module_name_in_model)
          param_name = '.'.join(module_name_in_model.split('.')[:-1])
          neuron_saliency = score_dict.get(param_name, None)
          
          
          # eps = 1e-11
          # eps=0
          # noise = torch.rand_like(neuron_saliency) * eps
          # neuron_saliency_for_threshold = 10e2*neuron_saliency + noise
          
          # if 'layer0_mlp_down' in module.name:print(neuron_saliency_for_threshold)

          n_pruned = int(len(neuron_saliency) * sparsity)
          topk_imp, _ = torch.topk(neuron_saliency, k=n_pruned, largest=False)
          threshold = topk_imp[-1]
          
            
          # if 'self_attn' in module_name_in_model:
          #   soft_mask_1d = torch.sigmoid((alpha/50) * (neuron_saliency_for_threshold - threshold - 1e-3))
          # elif 'mlp' in module_name_in_model:
          #   soft_mask_1d = torch.sigmoid(alpha * (neuron_saliency_for_threshold - threshold - 1e-3))
          if 'self_attn' in module_name_in_model:
            soft_mask_1d = torch.sigmoid((alpha/50) * (neuron_saliency - threshold))
          elif 'mlp' in module_name_in_model:
            soft_mask_1d = torch.sigmoid(alpha * (neuron_saliency - threshold))
          
          # print(soft_mask_1d.shape)
          
          if 'self_attn' in module_name_in_model:
            soft_mask_1d = torch.cat(
                      [torch.tensor([i for j in range(head_dim)])
                      for i in soft_mask_1d], 0)
            
            if module_name_in_model.split('.')[-1] in ['q_proj', 'o_proj']:
              soft_mask_1d = torch.cat(
                      [torch.tensor([i for j in range(kv_groups)])
                      for i in soft_mask_1d], 0)
              
          weight_mask = None
          bias_mask = None
              
          if module_name_in_model.split('.')[-1] in ['gate_proj', 'up_proj', 'q_proj', 'k_proj', 'v_proj']:
            weight_mask = soft_mask_1d.unsqueeze(1)  # a.k.a. row-wise mask
            # if module_name_in_model.split('.')[-1] in ['gate_proj', 'up_proj']:continue
            if module_name_in_model.split('.')[-1] in ['q_proj', 'k_proj', 'v_proj']:bias_mask = soft_mask_1d
            
          elif module_name_in_model.split('.')[-1] in ['down_proj', 'o_proj']:
            weight_mask = soft_mask_1d.unsqueeze(0)  # a.k.a. column-wise mask
            

          # print(f"Module: {module_name_in_model}, Mask shape: {structured_mask.shape}, Sparsity: {1 - structured_mask.mean().item()}, Smallest value: {structured_mask.min().item()}, Average value: {structured_mask.mean().item()}, Largest value: {structured_mask.max().item()}")
          # mask_dict[module.name] = structured_mask
          
          
          
          mask_dict[module.name] = {'weight': weight_mask, 'bias': bias_mask}


  # print(mask_dict)
  def debug_mask(mask):
    # logits = α * (x - (threshold + margin))
    # logits = alpha * (neuron_saliency - (threshold + margin))
    # mask = torch.sigmoid(logits)
    
    mask = mask.flatten()
    
    # 打印统计
    ones = (mask > 0.9).sum().item()
    zeros = (mask < 0.1).sum().item()
    halfs = ((mask >= 0.49) & (mask <= 0.51)).sum().item()
    # others = mask.numel() - (ones + zeros + halfs)

    print("=== Mask Debug ===")
    # print("Mask values:", mask.tolist())
    print(f"Total: {mask.numel()}")
    print(f"≈1 的个数 (>0.9): {ones}")
    print(f"≈0 的个数 (<0.1): {zeros}")
    print(f"≈0.5 的个数 (0.49~0.51): {halfs}")
    # print(f"介于 0/1 之间的个数: {others}")
    print("==================")
    # return mask
  
  # for key in mask_dict:
  #   print(key)
  #   # print(mask_dict[key])
  #   debug_mask(mask_dict[key]['weight'])
  #   try:
  #     debug_mask(mask_dict[key]['bias'])
  #   except:pass
  
  return mask_dict

def main(args):
  
  temperature = args.temperature
  noise_level = args.noise_level
  
  alpha = 50
  initial_alpha = alpha  
  final_alpha = 10e3
  
  
  
  global_step = 0
      
  wandb.init(
    project="calibration-selection", # Name of your project
    name=f"{args.base_model.split('/')[-1]}_lr_{args.lr}_noise_level_{noise_level}_temperature_{temperature}", # A descriptive name for this run
    config={
        "learning_rate": args.lr,
        "architecture": args.base_model,
        "epochs": args.iterative_steps,
        "sparsity_ratio": args.sparsity,
        "noise_level": noise_level, # Your noise level
        "u_temp": temperature, 
    }
  )
  
  
  
  
  print("Using noise_level:", noise_level)
  print("Using temperature:", temperature)
  
  tokenizer = AutoTokenizer.from_pretrained(args.base_model)
  
  
  example_prompts = get_loaders_(
                                name=args.data,
                                nsamples=args.num_examples,
                                seqlen=args.seq_len,
                                tokenizer=tokenizer,
                                idx=args.data_idx,
                                select_method='plain',
                                seed=args.seed).to(args.device)
  
  
  if example_prompts is None:
    print("No examples found")
    return
  
  
  train_data = get_loaders_(
                      name='alpaca',
                      nsamples=args.train_samples,
                      seqlen=args.seq_len,
                      tokenizer=tokenizer,
                      seed=42,
                      idx=[],
                      train=True)
  
  batch_size = args.batch_size
  train_loader = DataLoader(train_data, batch_size=batch_size)
  
  u = nn.Parameter(torch.ones(args.num_examples, device=args.device) / args.num_examples)
  
  if 'llama' in args.base_model.lower():
    model_config = LlamaConfig.from_pretrained(args.base_model,up=True)
    model = LlamaForCausalLM.from_pretrained(
      args.base_model,
      config=model_config,
      low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
      device_map='auto',
      torch_dtype=torch.bfloat16,
      attn_implementation="eager"
    )
  elif 'qwen' in args.base_model.lower():
    
    model_config = Qwen2Config.from_pretrained(args.base_model,up=True)
    model = Qwen2ForCausalLM.from_pretrained(
      args.base_model,
      config=model_config,
      low_cpu_mem_usage=True if args.torch_version >=1.9 else False,
      device_map='auto',
      torch_dtype=torch.bfloat16,
      attn_implementation="eager"
    )
  
  optimizer = torch.optim.AdamW(params = [u], lr=args.lr, betas=(0.9, 0.998), eps=1e-08, weight_decay=0.001)


  
  num_u_training_epochs = args.iterative_steps
  calib_batch_size = args.batch_size
  num_batches_per_epoch = (args.train_samples + calib_batch_size - 1) // calib_batch_size
  total_training_steps = num_batches_per_epoch * num_u_training_epochs

  warmup_steps = int(total_training_steps * 0.1)
  
  warmup_scheduler = LinearLR(
      optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
  )

  main_scheduler = CosineAnnealingLR(
      optimizer, T_max=(total_training_steps - warmup_steps), eta_min=0.0
  )

  scheduler = SequentialLR(
      optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
  )

  alpha_scheduler = AlphaScheduler(initial_alpha, final_alpha, total_training_steps)


  
  
  model.eval()
  
  
  before_pruning_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
  if args.test_before_prune:
    ppl = PPLMetric(model, tokenizer, ['wikitext', 'ptb', 'lambada'], args.max_seq_len, device=args.eval_device)
    print("PPL before pruning: {}".format(ppl))
  
  if args.prune:
    
    print("Start pruning with {} and {}".format(args.prune_type, args.data))    
    print("Start Backwarding in iterative steps = {}...".format(1))
  
    for epoch in range(args.iterative_steps):
      
      for data in tqdm(train_loader, desc="Training on calibration data"):

          
        if args.prune_type in ['NIRVANA', 'Pruner']:        
          cal_losses = []
          for i in range(args.num_examples):
              out = model(example_prompts[i].unsqueeze(0), labels=example_prompts[i].unsqueeze(0))
              loss = out[0]
              cal_losses.append(loss)
          cal_losses = torch.stack(cal_losses)
          
        
        
        
        L_cal = torch.sum(F.softmax(u/temperature, dim=0) * cal_losses)

        named_params = list(model.named_parameters())  # [(name, param), ...]
        
        grads = torch.autograd.grad(L_cal,
                                    [p for _, p in named_params], 
                                    create_graph=True)
        grad_dict = {name: g for (name, _), g in zip(named_params, grads)}

        
        
        alpha = alpha_scheduler.get_alpha()
        
        
        
        mask_dict = compute_structured_mask_dict(model, 
                                                  grad_dict,
                                                  sparsity=args.sparsity,
                                                  alpha=alpha,
                                                  method=args.prune_type
                                                  )
        
        
        data = data.to(args.device)
        inputs_embeds = model.model.embed_tokens(data)
        inputs_embeds.retain_grad()

        loss_grad = model(inputs_embeds=inputs_embeds, labels=data, mask_dict=mask_dict)[0]
        loss_grad.backward(retain_graph=True)

        
        # Uniform noise
        noise = (torch.rand_like(inputs_embeds) * 2 - 1) * noise_level
        perturbed_inputs_embeds = inputs_embeds + noise
        
          
          
        outputs = model(
            inputs_embeds=perturbed_inputs_embeds, 
            labels=data, 
            mask_dict=mask_dict
        )
        # reconstruction_loss = outputs[0]# + entropy
        L_target = outputs[0]
        L_target.backward(retain_graph=True)
        
        torch.nn.utils.clip_grad_norm_([u], max_norm=1.0)
        
        assert(u.grad is not None)
        
        log_dict = {
            "total_loss": L_target.item(),
            "u_grad_norm": u.grad.abs().mean().item(),
            "alpha": alpha,
        }
        
        u_probs = torch.softmax(u / temperature, dim=0)
        for i, prob in enumerate(u_probs):
          log_dict[f"u_prob_{i}"] = prob.item()
        
        wandb.log(log_dict, step=global_step)
        
        
        optimizer.step()
        
        scheduler.step()
        alpha_scheduler.step()

        optimizer.zero_grad()
        
          
        
        
        if (global_step) % 50==0:
          
          mask_dict = compute_structured_mask_dict(model, 
                                                      grad_dict,
                                                      sparsity=args.sparsity,
                                                      method=args.prune_type,
                                                      alpha=10e20, 
                                                      )
          
          # print(mask_dict)
          
          ppl = PPLMetric(model, tokenizer, ['wikitext', 'ptb', 'lambada'], args.max_seq_len, device=args.eval_device, mask_dict=mask_dict)
          print("PPL after pruning: {}, AVG {:.4f}".format(ppl, sum(ppl.values()) / len(ppl)))
          print("Current u:", F.softmax(u/temperature, dim=0))
      
      
      
        global_step +=1
      
    # Apply the hard mask
    if args.prune_type in ['NIRVANA', 'Pruner']:        
      cal_losses = []
      for i in range(args.num_examples):
          out = model(example_prompts[i].unsqueeze(0), labels=example_prompts[i].unsqueeze(0))
          loss = out[0]
          cal_losses.append(loss)
      cal_losses = torch.stack(cal_losses)
      
    
    
    L_cal = torch.sum(F.softmax(u/temperature, dim=0) * cal_losses)
      
      
      
    named_params = list(model.named_parameters())  # [(name, param), ...]
    grads = torch.autograd.grad(L_cal,
                                [p for _, p in named_params], 
                                create_graph=True)
    grad_dict = {name: g for (name, _), g in zip(named_params, grads)}

    
    
    mask_dict = compute_structured_mask_dict(model, 
                                              grad_dict,
                                              sparsity=args.sparsity,
                                              method=args.prune_type,
                                             alpha=10e20, 
                                            )  
      
    print("Final u:", F.softmax(u/temperature, dim=0))
    
    
    
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
      



  
  
  if args.test_after_prune:
    ppl = PPLMetric(model, tokenizer, ['wikitext', 'ptb','lambada'], args.max_seq_len, device=args.eval_device, mask_dict=mask_dict)
    print("PPL after pruning: {}, AVG {:.4f}".format(ppl, sum(ppl.values()) / len(ppl)))
      
  if args.save_model:
    
    import time, random

    torch.save(F.softmax((u/temperature), dim=0).detach().cpu().to(torch.float64),
           f"{args.base_model.split('/')[-1]}-{int(time.time()*1000)}-final_u.pt")
               




 

main(args)