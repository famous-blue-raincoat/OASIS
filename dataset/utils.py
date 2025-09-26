from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import torch

import random
import numpy as np

import json
from tqdm import tqdm



def get_bookcorpus_packed(tokenizer, n_samples, seq_len, seed=0, train=False, idx=-1, verbose=False, select_method=None):
    random.seed(seed)
    
    print("Using seed", seed, ", data method", select_method)
    
    verbose=True
    # if select_method:
    if select_method=='filter':
  
      
      tokenized_samples=[]
      
      # file_path = f"./dataset/{tokenizer.name_or_path.split('/')[-1]}_bookcorpus_filter.json"
      file_path = f"./dataset/{tokenizer.name_or_path.split('/')[-1]}_bookcorpus.json"
      
      # print(file_path)
      
      with open(file_path, "r", encoding="utf-8") as f:
          i_list = json.load(f)
      
      for idx, data in enumerate(i_list):
          if len(tokenized_samples) >= n_samples:break
          tokenized_sample = tokenizer(data['text'], return_tensors='pt')
          if tokenized_sample.input_ids.shape[1] - seq_len < 0:continue
          i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)  
          
          tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
          if verbose:print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
      
      return torch.cat(tokenized_samples, dim=0)

    else:
      
      ds = load_dataset('bookcorpus', split='train', trust_remote_code=True)
      windows = []
      
      
      traindata = ds.filter(lambda x: len(x["text"]) > 1024)
      # traindata = ds.filter(lambda x: len(x["text"]) > 2048)
      if select_method=='entropy':
        loaded_indices = torch.load(f'/work/nvme/beeo/Cali_sele/nirvana/Llama/bookcorpus_entropy_indices_{"_".join(tokenizer.name_or_path.split("/"))}_L1024_k1024.pt')
        # loaded_indices = torch.load(f'/work/nvme/beeo/Cali_sele/nirvana/Llama/bookcorpus_entropy_indices_meta-llama_{tokenizer.name_or_path.split('/')[-1]}_L1024_k1024.pt')
        traindata = traindata.select(loaded_indices.tolist())

        # loaded_indices = torch.load('/home/mai10/Cali_select/bookcorpus_entropy_indices_meta-llama_Llama-3.2-1B_L1024_k1024.pt')
      
      for text in traindata['text']:
          ids = tokenizer(text, add_special_tokens=False).input_ids
          if not ids: 
              continue
          if tokenizer.eos_token_id is not None:
              ids = ids + [tokenizer.eos_token_id]

          if len(ids) >= seq_len:
              offset = random.randint(0, min(seq_len-1, len(ids)-seq_len))
          else:
              continue

          stride = seq_len
          for start in range(offset, len(ids) - seq_len + 1, stride):
              windows.append(torch.tensor(ids[start:start+seq_len], dtype=torch.long))
              if len(windows) > 2:  # 小缓冲
                  break

      if len(windows) < n_samples:
          print(f"[warn] only {len(windows)} windows available.")
          n_samples = len(windows)

      # print(len(windows))
      
      if idx and len(idx) > 0 and not train:
        samples = torch.stack([windows[i] for i in idx], dim=0)
        # print("Using idx", idx)
        if verbose:
          for i in idx:
              print('sample', i, tokenizer.decode(windows[i]))
      
      else:
        # print("random")
        indices = random.sample(range(len(windows)), n_samples)
        samples = torch.stack([windows[i] for i in indices], dim=0)
        if verbose:
            for i in indices:
              print(tokenizer.decode(windows[i]))

      # if verbose:
      #     print("sample[0]:", tokenizer.decode(samples[0]))
      
      print("Data sample shape:", samples.shape)
      

    
        
    return samples

def get_alpaca_(tokenizer, n_samples, seq_len, idx=-1, verbose=False, seed=0, train=False):
    random.seed(seed)
    tokenized_samples, history = [], []


    print("[Calibration data alpaca] samples:", n_samples, "seq_len:", seq_len)
  
    traindata = load_dataset(
        'tatsu-lab/alpaca', split='train', trust_remote_code=True
    )
    
    # verbose=True
    
    # print(seq_len)
    
    # if idx>=0: 
    #   traindata = traindata.filter(lambda x: len(x["text"]) > 1024)
    # else:
    #   traindata = traindata.filter(lambda x: len(x["text"]) > seq_len)
    
    
    
    
    if idx>=0:       
        # print('idx', idx)
        tokenized_sample = tokenizer(traindata[idx]['text'], return_tensors='pt')
        if tokenized_sample.input_ids.shape[1] - seq_len < 0:
          print('too short')
          print(traindata[idx]['text'])
          print(tokenized_sample.input_ids)
          return None
        i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)
        tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])        
        if verbose:  
          print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
    
    else:
        for _ in range(n_samples):
            while True:
                i = random.randint(0, len(traindata) - 1)
                tokenized_sample = tokenizer(traindata[i]['text'], return_tensors='pt')
                # print(tokenized_sample.input_ids.shape[1])
                if tokenized_sample.input_ids.shape[1] >= seq_len and i not in history:
                    history.append(i)
                    break
            i = random.randint(0, tokenized_sample.input_ids.shape[1] - seq_len)        
            if verbose:
              print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
            tokenized_samples.append(tokenized_sample.input_ids[:, i:i+seq_len])
        # print(tokenizer.decode(tokenized_sample.input_ids[:, i:i+seq_len][0]))
        
          
    return torch.cat(tokenized_samples, dim=0)

 
def get_loaders_(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None, train=False, idx=-1, select_method=None):
    # print(nsamples)
    # print("data seed:", seed, 'seqlen', seqlen)
    if "bookcorpus" in name:
        return get_bookcorpus_packed(tokenizer=tokenizer, n_samples=nsamples, seq_len=seqlen, seed=seed, train=train, idx=idx, select_method=select_method)
    elif "alpaca" in name:
        return get_alpaca_(tokenizer=tokenizer, n_samples=nsamples, seq_len=seqlen, seed=seed, train=train)