import torch
from peft import PeftModel, PeftConfig
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 1. Load trained reward model
model_path = "output/cm/checkpoint-27714"
config = PeftConfig.from_pretrained(model_path)
base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

model.eval()
model.cuda()  # or device='auto'

dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="test")

# evaluation function
def score(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
        cost = outputs.logits.squeeze().item()
    return cost

# compute win-rate
correct = 0
ties = 0
for example in tqdm(dataset):
    prompt = example['prompt']
    better = example[f"response_{example['safer_response_id']}"]
    worse = example[f"response_{1 - example['safer_response_id']}"]

    better_score = score(f"User: {prompt}\nASSISTANT: {better}")
    worse_score = score(f"User: {prompt}\nASSISTANT: {worse}")

    if better_score > worse_score:
        correct += 1
    if better_score == worse_score:
        ties += 1

accuracy = correct / (len(dataset)-ties)
print(f"Cost model accuracy: {accuracy:.2%}")