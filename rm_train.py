from trl import RewardConfig, RewardTrainer
from datasets import load_dataset, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import Dict
from peft import LoraConfig, get_peft_model, TaskType

def convert_safe_rlhf_to_trl_format(example: Dict) -> Dict:
    better_response_id = example['better_response_id']
    prompt = example['prompt']
    better_response = example[f"response_{better_response_id}"]
    worse_response = example[f"response_{1-better_response_id}"]
    return {
        "prompt": prompt,
        "chosen": better_response,
        "rejected": worse_response
    }

tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForSequenceClassification.from_pretrained(
    "huggyllama/llama-7b", num_labels=1
)
model.config.pad_token_id = tokenizer.pad_token_id

lora_config = LoraConfig(
    task_type = TaskType.SEQ_CLS, # sequence classification
    r = 16, # rank
    lora_alpha = 32,
    lora_dropout = 0.05,
    bias = "none",
    target_modules = ["q_proj", "v_proj"],  # Llama targets
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() 

raw_dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
dataset = Dataset.from_list([convert_safe_rlhf_to_trl_format(example) for example in raw_dataset])

training_args = RewardConfig(output_dir="output/rm_llama_7b", per_device_train_batch_size=4)
trainer = RewardTrainer(
    args=training_args,
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
)
trainer.train()