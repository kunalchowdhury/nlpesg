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
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger("transformers")
logger.info("INFO")
logger.warning("WARN")
logging.get_logger("haystack").setLevel(logging.INFO)


def populate_training_parameters(data):
    model_name = 'distilbert-base-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = 384
    doc_stride = 128

    data["question"] = [q.lstrip() for q in data["question"]]
    data["context"] = [c.lstrip() for c in data["context"]]
    tokenized_data = tokenizer(
        data["question"],
        data["context"],
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    overflow_to_sample_mapping = tokenized_data.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_data.pop("offset_mapping")
    tokenized_data["start_positions"] = []
    tokenized_data["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        input_ids = tokenized_data["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        sequence_ids = tokenized_data.sequence_ids(i)
        sample_index = overflow_to_sample_mapping[i]
        answers = data["answers"][sample_index]
        if len(answers["answer_start"]) == 0:
            tokenized_data["start_positions"].append(cls_index)
            tokenized_data["end_positions"].append(cls_index)
        else:
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])
            token_start_idx = 0
            while sequence_ids[token_start_idx] != 1:
                token_start_idx += 1
            token_end_idx = len(input_ids) - 1
            while sequence_ids[token_end_idx] != 1:
                token_end_idx -= 1
            if not (
                    offsets[token_start_idx][0] <= start_char
                    and offsets[token_end_idx][1] >= end_char
            ):
                tokenized_data["start_positions"].append(cls_index)
                tokenized_data["end_positions"].append(cls_index)
            else:
                while (
                        token_start_idx < len(offsets)
                        and offsets[token_start_idx][0] <= start_char
                ):
                    token_start_idx += 1
                tokenized_data["start_positions"].append(token_start_idx - 1)
                while offsets[token_end_idx][1] >= end_char:
                    token_end_idx -= 1
                tokenized_data["end_positions"].append(token_end_idx + 1)
    return tokenized_data


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
        self.model = self.model()

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
            questions = get_questions(result)
            answers, contexts, indexes = get_context_answer(result)
            for i in range(len(questions)):
                m = {'text': answers[i]}
                l = [indexes[i]]
                m['answer_start'] = l
                self.df.loc[i] = [uuid.uuid4().hex, contexts[i], questions[i], m]
            print_questions(result)

    def prepare_test_validation_datasets(self):
        self.parse_file()
        # d = Dataset.from_pandas(self.df) #.train_test_split(test_size=0.1)
        # 90% train, 10% test + validation
        train_testvalid = Dataset.from_pandas(self.df).train_test_split(test_size=0.1)
        # Split the 10% test + valid in half test, half valid
        test_valid = train_testvalid['test'].train_test_split(test_size=0.5)
        # gather everyone if you want to have a single DatasetDict
        datasets = DatasetDict({
            'train': train_testvalid['train'],
            'test': test_valid['test'],
            'validation': test_valid['train']})
        # datasets = Dataset.from_pandas(self.df, split='train')
        # datasets = Dataset.from_pandas(df, split=NamedSplit('train'))
        # datasets = Dataset.from_pandas(df, split=datasets.Split.TRAIN)
        print(datasets["train"][0])
        tokenized_datasets = datasets.map(populate_training_parameters, batched=True,
                                          remove_columns=datasets["train"].column_names, num_proc=3, )
        train_set = tokenized_datasets["train"].with_format("numpy")[
                    :]  # Load the whole dataset as a dict of numpy arrays
        validation_set = tokenized_datasets["validation"].with_format("numpy")[:]
        return train_set, validation_set

    def model(self):
        optimizer = keras.optimizers.Adam(learning_rate=5e-5)
        keras.mixed_precision.set_global_policy("mixed_float16")
        model_checkpoint = 'distilbert-base-cased'
        model = TFAutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
        model.compile(optimizer=optimizer)
        train_set, validation_set = self.prepare_test_validation_datasets()
        model.fit(train_set, validation_data=validation_set, epochs=2)
        print("Fitting complete")
        return model

    def answer(self, context, question):
        model_checkpoint = 'distilbert-base-cased'
        tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
        inputs = tokenizer([context], [question], return_tensors="np")
        outputs = self.model(inputs)
        start_position = tf.argmax(outputs.start_logits, axis=1)
        end_position = tf.argmax(outputs.end_logits, axis=1)
        print(int(start_position), int(end_position[0]))
        answer = inputs["input_ids"][0, int(start_position): int(end_position) + 1]
        response = tokenizer.decode(answer)
        print(response)
        return response


# it is needed.

def main():
    esgnlp = ESGNLP("sample1.pdf")
    print("Model fitting complete")
    esgnlp.answer("Does your company have an ESG score or rating? Yes or No ", "Does your company have an ESG score or "
                                                                             "rating?")


if __name__ == "__main__":
    main()
