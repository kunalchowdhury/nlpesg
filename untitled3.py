# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:53:11 2024

@author: kunal
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
 
tokenizer_name = "ESGBERT/GovernanceBERT-governance"
model_name = "ESGBERT/GovernanceBERT-governance"
 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, max_len=512)
 
#pipe = pipeline("text-classification", model=model, tokenizer=tokenizer) # set device=0 to use GPU
 
# See https://huggingface.co/docs/transformers/main_classes/pipelines#transformers.pipeline
#print(pipe("An ethical code has been issued to all Group employees.", padding=True, truncation=True))
pipe=pipeline('question-answering', model=model, tokenizer=tokenizer)
print(pipe(question="What is CDP", context=text, top_k=3))
