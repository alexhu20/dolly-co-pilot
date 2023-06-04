model_path = '/Users/menghenghu/workspace/dolly/.trained_model/'

from transformers import pipeline
import torch

nlp = pipeline("question-answering", model=model_path, tokenizer=model_path)

# Provide a context and question for the model
context = "seeing error messages: Failed to update document in cosmos with precondition statusCode=PreconditionFailed and Response status code does not indicate success: PreconditionFailed (412)"
question = "location-ne-prd-exceptions-failures-logalertv2"

# Use the fine-tuned model to answer the question
result = nlp(question=question, context=context)
print(result)

# Print the answer
print("Question:", question)
print("Answer:", result["answer"])