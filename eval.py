from datasets import load_dataset 
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig, TrainingArguments, Trainer 
import torch
import time 
import evaluate 
import pandas as pd 
import numpy as np 

from utils import *
import evaluate

device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device('cpu'))

hf_data_name = 'knkarthick/dialogsum'
dataset = load_dataset(hf_data_name, cache_dir='./data')

# print(dataset)

saved_model_dir = f'./dialogue-summary-training'

model_name = 'google/flan-t5-base'

model = AutoModelForSeq2SeqLM.from_pretrained(saved_model_dir, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./models')

dialogues = dataset['test'][0:10]['dialogue']
human_baseline_summaries = dataset['test'][0:10]['summary']

instruct_model_summaries = []

for _, dialogue in enumerate(dialogues):
    prompt = f"""
        Summarize the following conversation.

        {dialogue}

        Summary: """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    instruct_model_outputs = model.generate(input_ids=input_ids, generation_config=GenerationConfig(max_new_tokens=200))
    instruct_model_text_output = tokenizer.decode(instruct_model_outputs[0], skip_special_tokens=True)
    instruct_model_summaries.append(instruct_model_text_output)
    
zipped_summaries = list(zip(human_baseline_summaries, instruct_model_summaries))
 
df = pd.DataFrame(zipped_summaries, columns = ['human_baseline_summaries', 'instruct_model_summaries'])
print(df.head)

rouge = evaluate.load('rouge')

instruct_model_results = rouge.compute(
    predictions=instruct_model_summaries,
    references=human_baseline_summaries[0:len(instruct_model_summaries)],
    use_aggregator=True,
    use_stemmer=True,
)

print('INSTRUCT MODEL:')
print(instruct_model_results)