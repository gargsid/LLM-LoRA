from datasets import load_dataset 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer 
import torch
import time, sys
import evaluate 
import pandas as pd 
import numpy as np 
from functools import partial

from utils import *

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print('device:', device)

hf_data_name = 'knkarthick/dialogsum'
dataset = load_dataset(hf_data_name, cache_dir='./data')
print(f'HF dataset--{hf_data_name} loaded')

model_name = 'google/flan-t5-base'
original_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./models', torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')
print(f'HF {model_name} loaded')

# dialogue = dataset['test'][200]['dialogue']
# summary = dataset['test'][200]['summary']

# prompt = f"""Summarize the conversation between triple backticks 

# ```
# {dialogue}
# ```

# Summary: """

# inputs = tokenizer(prompt, return_tensors='pt')
# # print('inputs',inputs)

# zeroshot_summary = tokenizer.decode(
#     original_model.generate(
#         inputs['input_ids'],
#         max_new_tokens=200,
#     )[0],
#     skip_special_tokens=True
# )

# dash_line = '-' * 100

# print(f'INPUT_PROMPT\n {prompt}')
# print(dash_line)

# print(f'GT-Summary: \n {summary}')
# print(dash_line)

# print(f'MODEL Summary: \n {zeroshot_summary}')
# print(dash_line)

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

output_dir = f'trained_models/dialogue-summary-training'

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
    model=original_model,
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
    original_model.generate(
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