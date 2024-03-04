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
from transformers import AutoTokenizer
import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModelForQuestionAnswering

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
# https://keras.io/examples/nlp/question_answering/
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
    def prepare_train_features(examples):
        model_checkpoint = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        max_length = 384  # The maximum length of a feature (question and context)
        doc_stride = (128  # The authorized overlap between two part of the context when splitting
                     )
        # Tokenize our examples with truncation and padding, but keep the overflows using a
        # stride. This results in one example possible giving several features when a context is long,
        # each of those features having a context that overlaps a bit the context of the previous
        # feature.
        examples["question"] = [q.lstrip() for q in examples["question"]]
        examples["context"] = [c.lstrip() for c in examples["context"]]
        tokenized_examples = tokenizer(
            examples["question"],
            examples["context"],
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )
    
        # Since one example might give us several features if it has a long context, we need a
        # map from a feature to its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original
        # context. This will help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")
    
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
    
        for i, offsets in enumerate(offset_mapping):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
    
            # Grab the sequence corresponding to that example (to know what is the context and what
            # is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
    
            # One example can give several spans, this is the index of the example containing this
            # span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]
            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])
    
                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != 1:
                    token_start_index += 1
    
                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != 1:
                    token_end_index -= 1
    
                # Detect if the answer is out of the span (in which case this feature is labeled with the
                # CLS index).
                if not (
                    offsets[token_start_index][0] <= start_char
                    and offsets[token_end_index][1] >= end_char
                ):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the
                    # answer.
                    # Note: we could go after the last offset if the answer is the last word (edge
                    # case).
                    while (
                        token_start_index < len(offsets)
                        and offsets[token_start_index][0] <= start_char
                    ):
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)
    
        return tokenized_examples    
    def prepare_test_validation_datasets(self):
        self.parse_file()
        datasets = Dataset.from_pandas(self.df)
        #datasets = Dataset.from_pandas(self.df, split='train')
        #datasets = Dataset.from_pandas(df, split=NamedSplit('train'))
        #datasets = Dataset.from_pandas(df, split=datasets.Split.TRAIN)
        print(datasets["train"][0])
        tokenized_datasets = datasets.map(self.prepare_train_features,batched=True,remove_columns=datasets["train"].column_names,num_proc=3,)
        train_set = tokenized_datasets["train"].with_format("numpy")[:]  # Load the whole dataset as a dict of numpy arrays
        validation_set = tokenized_datasets["validation"].with_format("numpy")[:]
        return train_set, validation_set
    def model(self):
        optimizer = keras.optimizers.Adam(learning_rate=5e-5)
        keras.mixed_precision.set_global_policy("mixed_float16")
        model_checkpoint = 'distilbert-base-cased'
        self.model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
        self.model.compile(optimizer=optimizer)
        train_set, validation_set=self.prepare_test_validation_datasets()
        self.model.fit(train_set, validation_data=validation_set, epochs=1)
        
    def test(self, context, question):
        model_checkpoint = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        inputs = tokenizer([context], [question], return_tensors="np")
        outputs = self.model(inputs)
        start_position = tf.argmax(outputs.start_logits, axis=1)
        end_position = tf.argmax(outputs.end_logits, axis=1)
        print(int(start_position), int(end_position[0]))
        answer = inputs["input_ids"][0, int(start_position) : int(end_position) + 1]
        print(answer)

   
# it is needed.
        
def main():
    esgnlp = ESGNLP("sample1.pdf")
    esgnlp.prepare_test_validation_datasets()


if __name__ == "__main__":
    main()
