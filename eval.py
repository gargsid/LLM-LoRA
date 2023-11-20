from datasets import load_dataset 
from datasets import concatenate_datasets

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import torch
import pandas as pd 
import numpy as np 
from functools import partial
import os, sys
from tqdm import tqdm 
import evaluate

import argparse
from utils import *
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print('device:', device)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='samsum')
parser.add_argument('--model_name', type=str, default='google/flan-t5-base')
parser.add_argument('--checkpoint_dir', type=str, default='trained_models')
parser.add_argument('--enable_lora', action='store_true')

args = parser.parse_args() 

hf_data_name = args.dataset_name
dataset = load_dataset(hf_data_name, cache_dir='./data')
print(f'HF dataset--{hf_data_name} loaded')

tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir='./models')

model_name = args.model_name
model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint_dir, torch_dtype=torch.bfloat16)
if args.enable_lora:
    lora_config = LoraConfig(
        r=32, # Rank
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM # FLAN-T5
    )
    model = get_peft_model(model, lora_config)

print(f'HF {model_name} loaded')
model.eval()

sample = dataset['test'][200]
dialogue = sample['dialogue']
human_summary = sample['summary']

prompt = """Summarize the conversation between triple backticks \n
```
{}
```\n
Summary: """

human_generated = []
model_generated = []

for sample in tqdm(dataset['test']):
    dialogue = sample['dialogue']
    human_summary = sample['summary']

    inputs = tokenizer(prompt.format(dialogue), return_tensors='pt')
    model_summary = tokenizer.decode(
        model.generate(
            input_ids=inputs['input_ids'],
            max_new_tokens=200,
        )[0],
        skip_special_tokens=True
    )

    human_generated.append(human_summary)
    model_generated.append(model_summary)
    # break

outputs = {
    'Human Summaries' : human_generated,
    'Model Generated Summaries' : model_generated,
}
df = pd.DataFrame(outputs)

output_csv_path = os.path.join(args.checkpoint_dir, 'generated_summaries.csv')
df.to_csv(output_csv_path)

'''
347
356
357
'''