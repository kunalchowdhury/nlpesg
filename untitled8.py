# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:45:22 2024

@author: kunal
"""
from haystack.telemetry import tutorial_running
from pprint import pprint
from tqdm.auto import tqdm
from haystack.nodes import QuestionGenerator, BM25Retriever, FARMReader
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.pipelines import (
    QuestionGenerationPipeline,
    RetrieverQuestionGenerationPipeline,
    QuestionAnswerGenerationPipeline,
)
from haystack.utils import launch_es, print_questions
import os
from subprocess import Popen, PIPE, STDOUT
import logging
from haystack.nodes import PDFToTextConverter
from pathlib import Path

def start_index(str1, str2):
    str1_words=str1.split()
    str2_words=str2.split()
    str1_indexes={}
    str2_indexes={}
    i=0
    for word in str1_words:
        str1_indexes[word]=i
        i=i+len(word)+1
    i=0
    for word in str2_words:
        str2_indexes[word]=i
        i=i+len(word)+1
    print(str1_indexes)
    print(str2_indexes)
    for key in str1_indexes:
        if key in str2_indexes:
            return str2_indexes[key]
    return -1    

def get_questions(result):
    result["queries"] #returns list of questions

answers=[]
def get_context_answer(result):
    for answer in result["answers"]:
        answers.append(print(answer[0].answer).strip())

text="""ESG investing exists within a broader spectrum of investing based on financial and social returns. On one
side of the spectrum, pure financial investment is pursued to maximise shareholder and debtholder value
through financial returns based on absolute or risk-adjusted measures of financial value. At best, it
assumes the efficiency of capital markets will effectively allocate resources to parts of the economy that
maximise benefits, and contributes more broadly to economic development. On the other side of the
spectrum, pure social “investing, such as philanthropy, seeks only social returns, such that the investor
gains from confirming evidence of benefits to segments or all of society, in particular related to
environmental or social benefits, including with regard to human and worker rights, gender equality. Social
impact investing seeks a blend of social return and financial return – but the prioritisation of social or
financial returns depends on the extent to which the investors are willing to compromise one for the other
in alignment with their overall objectives."""
docs= [{"content": text}]
question_generator = QuestionGenerator()

document_store = ElasticsearchDocumentStore()
#document_store.write_documents(docs)

reader = FARMReader("deepset/roberta-base-squad2")
qag_pipeline = QuestionAnswerGenerationPipeline(question_generator, reader)
for idx, document in enumerate(tqdm(document_store)):
 
     print(f"\n * Generating questions and answers for document {idx}: {document.content[:100]}...\n")
     result = qag_pipeline.run(documents=[document])
     print_questions(result)


#print(start_index("ESG investing exists within a broader spectrum of what?", text))


        