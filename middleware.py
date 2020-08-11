from anserini_retriever import *
from gpt_worker import *

"""
THIS FILE MAINLY CONTAINS FUNCTIONS THAT COMMUNICATE BETWEEN MAIN AND RETRIEVER/WORKER
"""


def getDocumentContent(CONF, docid):
    # Simple method to get the text content of the document from the docid
    retriever = AnseriniRetriever(CONF)
    doc = retriever.getDoc(docid)
    content = doc.raw()
    return content


def getPredictionScore(document: str, query: str):
    # Get the prediction score from the GPT-2 model
    worker = GPT2()
    queryTokens = query.split(" ")

    prob = worker.prediction(document, queryTokens)
    print(prob)
