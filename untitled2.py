# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:24:01 2024

@author: kunal
"""

from transformers import pipeline

from datasets import load_dataset

pipe = pipeline("text-generation", model='ESGBERT/GovernanceBERT-governance', truncation=True)
print(pipe("What is ESG", min_length=50, max_length=100, truncation=True)[0]['generated_text'])