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

## Sample Results

Let's look at couple of dialogues and compare summaries from different sources -- human generated, full-finetuned FlanT5, LoRA FlanT5. 

### Dialogue-1

```
David: Morning Russ. Have you seen the report I emailed yesterday?

Russ: Hi David. Well received thank you. But I haven't read it yet.

David: Is there anything you'd like me to do right now?

Russ: I'll take a look at the report in a moment and will send you remarks if I have any.

David: Sounds good. I guess I'll just answer some emails.

Russ: Please do. I should be done by midday with the report.
```

**Human Summary:**
```
Russ received David's report but hasn't read it yet.
```

**Full-Finetuned Model Summary:** 
```
David has received the report he emailed yesterday. Russ will take a look at it and send David remarks. David will be done by midday with the report.
```
**LoRA Model Summary:**
```
David has emailed Russ a report. Russ will read it and send David some comments.
```

### Dialogue-2

```
Millie: Heeey I’m sick I won’t come today

Sal: I’m sorry! Get better soon :*

Millie: <3
```

**Human Summary:**
```
Millie is sick, so she won't come today.
```
**Full-Finetuned Model Summary:** 
```
Millie is sick and won't come today.
```
**LoRA Model Summary:**
```
Millie is sick and won't come today. 
```

We can see that finetuning only 1.4% parameters using LoRA is giving really good results which are closer to the results of the full-finetuned model. Using PEFT methods can save us a lot of time without compromising on the quality of the model. 

## ROUGE Scores

We can also see that the ROUGE-scores are also very close for both the methods.

| Metric  | Full-Finetuning  | LoRa  |
|---|---|---|
| ROUGE-1  | 0.478  | 0.474  |
| ROUGE-2  | 0.231  | 0.227  |
| ROUGE-L  | 0.395  | 0.392  |



# Acknowledgements

- Coursera's Introduction to Generative AI Course

- Blog: https://www.philschmid.de/fine-tune-flan-t5


