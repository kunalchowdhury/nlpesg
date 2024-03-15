# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:07:17 2024

@author: kunal
"""

from haystack import Document
from haystack import Pipeline
from haystack.components.converters.pypdf import PyPDFToDocument
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.readers import ExtractiveReader
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.writers import DocumentWriter
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.preprocessors import TextCleaner
import fitz


def remove_images(input_pdf, output_pdf):
    doc = fitz.open(input_pdf)
    for page in doc:
        img_list = page.get_images()
        for img in img_list:
            page.delete_image(img[0])
    doc.save(output_pdf)


def preprocess(files):
    converter = PyPDFToDocument()
    results = converter.run(sources=files)
    documents = results["documents"]
    cleaner = DocumentCleaner()
    results = cleaner.run(documents=documents)
    doc_splitter = DocumentSplitter(split_by="sentence", split_length=10)
    results = doc_splitter.run(results['documents'])
    cleaner = TextCleaner(convert_to_lowercase=True, remove_punctuation=True, remove_numbers=False,
                          remove_regexps=["\n\n", "\r"])
    docs = []
    for result in results['documents']:
        cur_doc = Document(content=''.join(cleaner.run(result.content)["texts"]).replace("\n", " "), meta=result.meta)
        docs.append(cur_doc)
        print(cur_doc)
    return docs


def create_document_store(documents, model):
    document_store = InMemoryDocumentStore()

    indexing_pipeline = Pipeline()

    indexing_pipeline.add_component(instance=SentenceTransformersDocumentEmbedder(model=model), name="embedder")
    indexing_pipeline.add_component(instance=DocumentWriter(document_store=document_store), name="writer")
    indexing_pipeline.connect("embedder.documents", "writer.documents")

    indexing_pipeline.run({"documents": documents})
    print("done running indexing pipeline")
    return document_store


def create_extractive_qa_pipeline(document_store, model):
    retriever = InMemoryEmbeddingRetriever(document_store=document_store)
    reader = ExtractiveReader()
    reader.warm_up()

    extractive_qa_pipeline = Pipeline()

    extractive_qa_pipeline.add_component(instance=SentenceTransformersTextEmbedder(model=model), name="embedder")
    extractive_qa_pipeline.add_component(instance=retriever, name="retriever")
    extractive_qa_pipeline.add_component(instance=reader, name="reader")

    extractive_qa_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    extractive_qa_pipeline.connect("retriever.documents", "reader.documents")
    return extractive_qa_pipeline


class ESGQA:
    def __init__(self, files):
        documents = preprocess(files)
        model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
        document_store = create_document_store(documents, model)
        self.extractive_qa_pipeline = create_extractive_qa_pipeline(document_store, model)

    def answer(self, query):
        try:
            ans = self.extractive_qa_pipeline.run(
                data={"embedder": {"text": query}, "retriever": {"top_k": 1}, "reader": {"query": query, "top_k": 1}}
            )
            print(ans)
            return ans
        except Exception as e:
            print(e)


def main():
    esg_qa = ESGQA(["sample1.pdf", "example.pdf"])
    print(esg_qa.answer("What are the various tools for green investment ? "))


if __name__ == "__main__":
    main()

# create_indexing_pipeline(preprocess(["sample1.pdf", "example.pdf"]))
