# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 19:08:39 2024

@author: kunal
"""

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering
import torch 
#import PyPDF2
#from Pypdf import PdfReader
#from PyPdf import PdfFileReader
import PyPDF2
import io
from tika import parser # pip install tika
from transformers import GPT2Config,LongformerConfig, GPT2Model
from transformers import GPT2Tokenizer
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
from transformers import AutoTokenizer, LongformerForMaskedLM

import spacy
nlp = spacy.load('en_core_web_sm')

def is_question(sent):
    d = nlp(sent)
    token = d[0] # gets the first token in a sentence
    if token.tag_ == "VERB" or token.tag_ == "VB" or token.tag_ == "VBD" or token.tag_ == "VBG" or token.tag_ == "VBN" or token.tag_ == "VBP" or token.tag_ == "VBZ" and token.dep_ == "ROOT": # checks if the first token is a verb and root or not
        return True
    for token in d: # loops through the sentence and checks for WH tokens
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
            return True
    return  False

def getPDFContent(filename):
    with open(filename, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            #page = reader.getPage(page_number)
            text = page.extract_text()
            #lines = text.split('\n')
            for line in io.StringIO(text):
                if "." in line and "www." not in line:
                    line=line.split(".")[1].strip()
                    print(line)
                print(f'line {line} is a question ->> {is_question(line)}')
        #print(lines)
#getPDFContent("sample1.pdf")
model_ckpt="ESGBERT/GovernanceBERT-governance"
#configuration = LongformerConfig()
model=AutoModelForQuestionAnswering.from_pretrained(model_ckpt)
#model = LongformerForMaskedLM.from_pretrained("ESGBERT/GovernanceBERT-governance")
tokenizer=AutoTokenizer.from_pretrained(model_ckpt)
#tokenizer = AutoTokenizer.from_pretrained("ESGBERT/GovernanceBERT-governance")

raw=parser.from_file('example.pdf')
text=raw['content']

pipe=pipeline('question-answering', model=model, tokenizer=tokenizer)
print(pipe(question="What is CDP", context=text, topk=3))
#pipe=pipeline('text-generation', model=model, tokenizer=tokenizer)
#print(pipe("What is ESG"))
