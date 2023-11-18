# Chat Dialogue Summarization using FlanT5 and LoRA

In this work, we investigated the effectiveness of FlanT5 model for summarizing dialogues. We used 3 strategies. We generated zero-shot summaries, summaries after fine-tuning the full model on a training dataset, and also fine-tuned the model using Parameter Efficient Fine Tuning method called LoRA (Low-Rank Approximation). 

In summary, the zero-shot summaries are not accurate and fine-tuning is necessary. However, full-finetuning of LLMs like FlanT5 is computationally effective and therefore, we also compared how PEFT methods like LoRA perform against full finetuning. We show that by using LoRA we only train 1.4% of the total parameters (total params: 250M) but save tremendous compute and training time and still achieving similar performance as the full finetuned model. 

## Setting up

Clone the repository using 

```
git clone https://github.com/gargsid/LLM-LoRA.git
```

Go to the working directory

```
cd LLM-LoRA
```

To replicate the working environmet, please use environment.yml file

```
conda env create -f environment.yml
```

Activate the newly created environment using

```
conda activate llm
```

To fine-tune the full model using a summarization datase like `samsum` or `knkarthick/dialogsum` using FlanT5 model use

```
python main.py --model_name=google/flan-t5-base --dataset_name=samsum
```

For doing PEFT training using LoRA, use `--enable_lora` flag with the above command

# Acknowledgements

- Coursera's Introduction to Generative AI Course

- Blog: https://www.philschmid.de/fine-tune-flan-t5


