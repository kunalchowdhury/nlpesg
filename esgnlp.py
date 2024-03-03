# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 16:07:17 2024

@author: kunal
"""
from pathlib import Path
import pandas as pd
import uuid

from pre_process import prepare_qs_n_as, get_questions, get_context_answer
import logging
# Imports needed to run this notebook

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
from haystack.nodes import PDFToTextConverter
from datasets import Dataset, DatasetDict
from elasticsearch import Elasticsearch

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

# https://haystack.deepset.ai/tutorials/13_question_generation
# https://docs.haystack.deepset.ai/docs/file_converters
# https://discuss.huggingface.co/t/from-pandas-dataframe-to-huggingface-dataset/9322/3
# https://colab.research.google.com/github/keras-team/keras-io/blob/master/examples/nlp/ipynb/question_answering.ipynb#scrollTo=KASsuiYLkTUd
# Before starting delete all Elastic indexes using POSTMAN -
# GET http://localhost:9200/document/_search?pretty=true&q=*:*
# And DELETE http://localhost:9200/document/_doc/965cecd0c9a5452296a701d6299d0b8d

class ESGNLP:
    def __init__(self, file):
        self.df = pd.DataFrame(
            columns=('id', 'context', 'question', 'answers'))
        self.file = file
        es = Elasticsearch('http://localhost:9200')
        resp = es.search(index="document", query={"match_all": {}})
        print('Purging Elasticsearch..')
        for doc_id in resp['hits']['hits']:
            print(doc_id['_id'])
            es.delete(index="document", id=doc_id['_id'])

    def parse_file(self):
        converter = PDFToTextConverter(
            remove_numeric_tables=True, valid_languages=["en"])
        docs = converter.convert(file_path=Path(self.file), meta=None)
        question_generator = QuestionGenerator()
        document_store = ElasticsearchDocumentStore()
        document_store.write_documents(docs)
        reader = FARMReader("deepset/roberta-base-squad2")
        qag_pipeline = QuestionAnswerGenerationPipeline(
            question_generator, reader)
        for idx, document in enumerate(tqdm(document_store)):
            print(
                f"\n * Generating questions and answers for document {idx}: {document.content[:100]}...\n")
            result = qag_pipeline.run(documents=[document])
            questions=get_questions(result)
            answers, contexts, indexes=get_context_answer(result)
            for i in range(len(questions)):
                self.df.loc[i]=[uuid.uuid4().hex, contexts[i], questions[i], "{'text' : ['" + answers[i] + "'], 'answer_start': [" + str(indexes[i]) +"]}"  ]
            print_questions(result)
    def get_dataset(self):
        self.parse_file()
        dataset = Dataset.from_pandas(self.df)
        #dataset = Dataset.from_pandas(df, split='train')
        #dataset = Dataset.from_pandas(df, split=NamedSplit('train'))
        #dataset = Dataset.from_pandas(df, split=datasets.Split.TRAIN)
        print(dataset)
        return dataset
def main():
    esgnlp = ESGNLP("sample1.pdf")
    esgnlp.get_dataset()


if __name__ == "__main__":
    main()
