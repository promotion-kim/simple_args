import sys
sys.path.append('/home/sjkim/srgtg')
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from datasets import load_dataset
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from peft import PeftModel, PeftConfig

rm_path = 'output/rm/checkpoint-27714'
rm_config = PeftConfig.from_pretrained(rm_path)
rm_base_model = AutoModelForSequenceClassification.from_pretrained(rm_config.base_model_name_or_path, num_labels=1, device_map='auto')
rm = PeftModel.from_pretrained(rm_base_model, rm_path)
tokenizer = AutoTokenizer.from_pretrained(rm_config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
rm.config.pad_token_id = tokenizer.pad_token_id
rm.eval()

cm_path = 'output/cm/checkpoint-27714'
cm_config = PeftConfig.from_pretrained(cm_path)
cm_base_model = AutoModelForSequenceClassification.from_pretrained(cm_config.base_model_name_or_path, num_labels=1, device_map='auto')
cm = PeftModel.from_pretrained(cm_base_model, cm_path)
cm.config.pad_token_id = tokenizer.pad_token_id
cm.eval()

sft_model_name = 'meta-llama/Llama-3.2-3B-Instruct'
sft = AutoModelForCausalLM.from_pretrained(
    sft_model_name,
    torch_dtype=torch.bfloat16,
    device_map='auto'
)
sft.eval()

def generate_args(prompt, k, w, max_length=512):

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(sft.device)
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = tokenizer.eos_token_id
    device = input_ids.device

    best_idx = None

    B = input_ids.size(0) # number of batches
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    while True:
        if best_idx is None:
            # first step: get entire prefix
            sft_step = sft(input_ids=input_ids, use_cache=True, return_dict=True)
            rm_step = rm(input_ids=input_ids, use_cache=True, return_dict=True)
        else:
            # just new token
            sft_step = sft(input_ids=best_idx, past_key_values=sft_cache, use_cache=True, return_dict=True)
            rm_step = rm(input_ids=best_idx, attention_mask=torch.ones_like(best_idx, dtype=torch.bool), past_key_values=rm_cache, use_cache=True, return_dict=True)

        sft_cache = sft_step.past_key_values
        rm_cache = rm_step.past_key_values

        with torch.no_grad():
            logits = sft_step.logits[:, -1] # size = (1, V)
            log_probs = F.log_softmax(logits, dim=-1) # size = (1, V)
            topk_log_probs, topk_tokens = torch.topk(log_probs, k, dim=-1)
        B, k = topk_tokens.size()
        # repeat prefix (B,1,L) -> (B,k,L)  
        prefix_repeat = input_ids.unsqueeze(1).repeat(1, k, 1) # (B, k, L)  
        cand_tokens = topk_tokens.unsqueeze(-1) # (B, k, 1) 
        candidates = torch.cat([prefix_repeat, cand_tokens], dim=-1).view(B*k, -1) # (B*k, L+1)
        with torch.no_grad():
            candidates = candidates.to(device)                           
            attention_mask = (candidates != pad_token_id).to(device)
            rm_out = rm(
                input_ids=candidates,
                attention_mask=attention_mask,
                return_dict=True,
                use_cache=True,
            )
            flat_rewards = rm_out.logits[:, 0] # (B * k)

            rewards = flat_rewards.view(B, k).to(device) # (B, k)
            #print(f"rewards: {rewards}")
            scores = topk_log_probs + w * rewards
        best_pos = scores.argmax(dim=-1)
        best_tokens = topk_tokens[torch.arange(B), best_pos]

        with torch.no_grad():
            just_finished = best_tokens.eq(eos_token_id)
            finished = finished | just_finished.to(device)
            add_tokens = best_tokens.clone()
            add_tokens[finished] = pad_token_id
            best_idx = add_tokens.unsqueeze(-1)  # (B, 1)
            input_ids = torch.cat([input_ids, best_idx], dim=-1)
        print(tokenizer.batch_decode(input_ids, skip_special_tokens = True))

        if input_ids.size()[1] == max_length or finished.all():
            break
    
    return input_ids

test_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")
test_dataset = test_dataset.select(range(100))

def collate_fn(batch):
    prompts = []
    better_answers = []
    worse_answers = []
    for example in batch:
        prompts.append(example['prompt'])
        better_response_id = example['better_response_id']
        better_answers.append(example[f'response_{better_response_id}'])
        worse_answers.append(example[f'response_{1-better_response_id}'])
    return {
        'prompt': prompts,
        'better_answer': better_answers,
        'worse_answer': worse_answers
        }

eval_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

device = sft.device
eos_token_id = tokenizer.eos_token_id
pad_token_id = tokenizer.pad_token_id
total = 0
helpfulness_wins = 0
safety_wins = 0
helpfulness_ties = 0
safety_ties = 0
max_length = 256
w = 1.0
k = 10

rows = []

for batch in tqdm(eval_loader):
    prompts = batch['prompt']
    better_answers = batch['better_answer']
    worse_answers  = batch['worse_answer']

    for prompt, better_answer, worse_answer in tqdm(zip(prompts, better_answers, worse_answers)):
        prompt_len = len(tokenizer(prompt)['input_ids'])
        
        input_ids_sft = sft.generate(
            tokenizer(prompt, return_tensors="pt").input_ids.to(device),
            max_new_tokens=max_length-prompt_len,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id
        )
    
        input_ids_args = generate_args(prompt=prompt, max_length=max_length, w=w, k=k)

        better_input_ids = tokenizer(prompt+better_answer, return_tensors="pt").input_ids.to(device)
        worse_input_ids = tokenizer(prompt+worse_answer, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            r_sft = rm(input_ids=input_ids_sft).logits[:, 0].item()
            r_args = rm(input_ids=input_ids_args).logits[:, 0].item()
            c_sft = cm(input_ids=input_ids_sft).logits[:, 0].item()
            c_args = cm(input_ids=input_ids_args).logits[:, 0].item()

        sft_answer = tokenizer.decode(input_ids_sft.squeeze(), skip_special_tokens=True)
        args_answer = tokenizer.decode(input_ids_args.squeeze(), skip_special_tokens=True)
        total += 1
        if r_args > r_sft:
            helpfulness_wins += 1
        elif r_args == r_sft:
            helpfulness_ties += 1
        
        if c_args < c_sft:
            safety_wins += 1
        elif c_args == c_sft:
            safety_ties += 1

        rows.append(
            {'prompt': prompt,
            'sft': sft_answer,
            'args': args_answer,
            'r_sft': r_sft,
            'r_args': r_args,
            'c_sft': c_sft,
            'c_args': c_args
            }
        )

        df = pd.DataFrame(rows)
        df.to_csv(f'/home/sjkim/rgtg/output/sft_vs_args_w_{w}_k_{k}.csv', index=False)

print(f"helpfulness win rate: {100*(helpfulness_wins)/(total-helpfulness_ties)}")
print(f"safety win rate: {100*(safety_wins)/(total-safety_ties)}")



