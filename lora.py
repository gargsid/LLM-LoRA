from datasets import load_dataset 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer 
import torch
import time 
import evaluate 
import pandas as pd 
import numpy as np 
from functools import partial

from utils import *

from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=32, # Rank
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
)

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print('device:', device)

hf_data_name = 'knkarthick/dialogsum'
dataset = load_dataset(hf_data_name, cache_dir='./data')
print(f'HF dataset--{hf_data_name} loaded')

model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./models', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')

peft_model = get_peft_model(original_model, lora_config)
print(f'HF PEFT {model_name} loaded')
print(print_number_of_trainable_params(peft_model))

# print(print_number_of_trainable_params(original_model))

def tokenize_function(example):
    start_prompt = 'Summarize the following conversation. \n\n'
    end_prompt = '\n\nSummary:'
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example['dialogue']]
    example['input_ids'] = tokenizer(prompt, padding='max_length', truncation=True, return_tensors='pt').input_ids
    example['labels'] = tokenizer(example['summary'], padding='max_length', truncation=True, return_tensors='pt').input_ids
    return example

print('Tokenizing dataset...')
tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(['id', 'topic', 'dialogue', 'summary'])
print('Tokenized Dataset:', tokenized_datasets)

output_dir = f'trained_models/dialogue-summary-peft-training'

rouge = evaluate.load('rouge')
def rouge_wrapper(eval_preds):
    preds, labels = eval_preds
    results = rouge.compute(preds,labels, use_aggregator=True, use_stemmer=True)
    return results

training_args = TrainingArguments(
    learning_rate=1e-5,
    num_train_epochs=50,
    weight_decay=0.01,
    evaluation_strategy = "steps",
    eval_steps = 500,
    # logging_steps=10,
    output_dir=output_dir,
    overwrite_output_dir=True,
    save_strategy='steps',
    save_steps=500,
    save_total_limit=1,
    load_best_model_at_end=True,
    disable_tqdm=True,
    # metric_for_best_model='rougeL',
    # greater_is_better=True,
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    # compute_metrics=rouge_wrapper
)

trainer.train() 

trainer.save_model(output_dir)
print(f'Saved to {output_dir}')

print(f'MODEL Summary (ZeroShot): \n {zeroshot_summary}')
print(dash_line)

fine_tuned_summary = tokenizer.decode(
    peft_model.generate(
        inputs['input_ids'].to(device),
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

print(f'MODEL Summary (Fine-Tune): \n {fine_tuned_summary}')
print(dash_line)

'''
1. eval based on val rouge-score instead of val loss.
2. save some sample preds at evaluation. 
'''