# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 18:20:26 2024

@author: kunal
"""
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
import torch 

#model_ckpt="nbroad/ESG-BERT"
model_ckpt="deepset/minilm-uncased-squad2"
tokenizer=AutoTokenizer.from_pretrained(model_ckpt)

#question="How much music can this hold?"
question="What is MP3"
context=""" An MP3 is about 1MB/minute so about 6000 hours depending on file size"""
inputs=tokenizer(question, context, return_tensors="pt")

model=AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
with torch.no_grad():
    outputs=model(**inputs)
print(outputs)

start_logits=outputs.start_logits
end_logits=outputs.end_logits

start_idx=torch.argmax(start_logits)
end_idx=torch.argmax(end_logits) + 1

answer_span=inputs["input_ids"][0][start_idx:end_idx]
answer=tokenizer.decode(answer_span)
print(f'Question :{question}')
print(f'Answer :{answer}')