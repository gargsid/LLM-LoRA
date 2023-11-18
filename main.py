from datasets import load_dataset 
from datasets import concatenate_datasets

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

import torch
import pandas as pd 
import numpy as np 
from functools import partial
import os

import evaluate
import nltk
from nltk.tokenize import sent_tokenize
nltk.download("punkt")

import argparse
from utils import *
from peft import LoraConfig, get_peft_model, TaskType

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))
print('device:', device)

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default='samsum')
parser.add_argument('--model_name', type=str, default='google/flan-t5-base')
parser.add_argument('--output_dir', type=str, default='trained_models/')
parser.add_argument('--enable_lora', action='store_true')

args = parser.parse_args() 

hf_data_name = args.dataset_name
dataset = load_dataset(hf_data_name, cache_dir='./data')
print(f'HF dataset--{hf_data_name} loaded')

tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir='./models')

model_name = args.model_name
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir='./models', torch_dtype=torch.bfloat16)
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
print(print_number_of_trainable_params(model))

sample = dataset['test'][200]
dialogue = sample['dialogue']
human_summary = sample['summary']

prompt = """Summarize the conversation between triple backticks \n
```
{}
```\n
Summary: """

inputs = tokenizer(prompt.format(dialogue), return_tensors='pt')
zeroshot_summary = tokenizer.decode(
    model.generate(
        input_ids=inputs['input_ids'],
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

# print(f'Input Prompt\n {prompt.format(dialogue)}')
# print(dash_line)

# print(f'GT-Summary: \n {human_summary}')
# print(dash_line)

# print(f'MODEL Summary: \n {zeroshot_summary}')
# print(dash_line)

# The maximum total input sequence length after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["dialogue"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_source_length = max([len(x) for x in tokenized_inputs["input_ids"]])
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization. 
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([dataset["train"], dataset["test"]]).map(lambda x: tokenizer(x["summary"], truncation=True), batched=True, remove_columns=["dialogue", "summary"])
max_target_length = max([len(x) for x in tokenized_targets["input_ids"]])
print(f"Max target length: {max_target_length}")

def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    inputs = ["summarize: " + item for item in sample["dialogue"]]

    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["summary"], max_length=max_target_length, padding=padding, truncation=True)

    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["dialogue", "summary", "id"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# Metric
metric = evaluate.load("rouge")

# helper function to postprocess text
def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in result.items()}
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result

# we want to ignore tokenizer pad token in the loss
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

save_dir = f'summarization_{args.dataset_name}'
if args.enable_lora:
    save_dir += f'_lora'
output_dir = os.path.join(args.output_dir, save_dir)

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    fp16=False, # Overflows with fp16
    learning_rate=5e-5,
    num_train_epochs=5,
    # logging & evaluation strategies
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rougeL",
    disable_tqdm=True,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    compute_metrics=compute_metrics,
)

# Start training
trainer.train()

trainer.evaluate()

dash_line = '-' * 25
# print(dash_line)

inputs = tokenizer(prompt.format(dialogue), return_tensors='pt')
fine_tuned_summary = tokenizer.decode(
    model.generate(
        inputs['input_ids'].to(device),
        max_new_tokens=200,
    )[0],
    skip_special_tokens=True
)

print(prompt.format(dialogue))
print(dash_line)

print('HUMAN Summary:')
print(human_summary)
print(dash_line)

print('Zeroshot Summary')
print(zeroshot_summary)
print(dash_line)

print('Finetuned model summary')
print(fine_tuned_summary)
