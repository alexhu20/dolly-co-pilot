# Dolly

This fine-tunes the [GPT-J 6B](https://huggingface.co/EleutherAI/gpt-j-6B) model on the [Alpaca](https://huggingface.co/datasets/tatsu-lab/alpaca) dataset using a Databricks notebook.  Please note that while GPT-J 6B is [Apache 2.0 licensed](https://huggingface.co/EleutherAI/gpt-j-6B), the Alpaca dataset is licensed under [Creative Commons NonCommercial (CC BY-NC 4.0)](https://huggingface.co/datasets/tatsu-lab/alpaca).

## Get Started Training

### Install dependency

```
pip install -r requirements_dev.txt
```

### Train the model

* Start a single-node cluster with node type having 8 A100 (40GB memory) GPUs (e.g. `Standard_ND96asr_v4` or `p4d.24xlarge`).

```bash
python3 run_training.py \
  --model_name_or_path BERT \
  --train_file {training_data.csv}\
  --test_file {test_data.csv} \
  --do_train \
  --per_device_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./.trained_model/ \
	--bf16 \
	--no_cuda \
	--overwrite_output_dir
```

## Load the data and answering some questions
```
model_path = '/Users/menghenghu/workspace/dolly/trained_model/'

from transformers import pipeline
import torch

nlp = pipeline("question-answering", model=model_path, tokenizer=model_path)

# Provide a context and question for the model
context = "seeing error messages: Failed to update document in cosmos with precondition statusCode=PreconditionFailed and Response status code does not indicate success: PreconditionFailed (412); Substatus: 0; ActivityId: 5110eb22-228f-4270-b79f-31ed80382011; Reason: "
question = "location service error"

# Use the fine-tuned model to answer the question
result = nlp(question=question, context=context)
print(result)

# Print the answer
print("Question:", question)
print("Answer:", result["answer"])
```

```python
model_path = '/path/to/checkpoint'
from training.generate import load_model_tokenizer_for_generate, generate_response
model, tokenizer = load_model_tokenizer_for_generate(model_path)
instruction='Write a tweet to introduce Dolly, a model to mimic ChatGPT.'
response = generate_response(instruction, model, tokenizer)
print(response)
```

(It is recommended to use `ipython` to interactively generate sentences to avoid loading models from disk again and again.)
