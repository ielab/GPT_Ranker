from anserini_retriever import *
from gpt_worker import *
import numpy

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
    score = numpy.sum(numpy.log(prob))
    return score


def rerankDocuments(RANKED_FILE_CONTENT, QUERY, CONF):
    # Construct the sorted re-ranked list
    count = 0
    rankedCollection = []
    total = 0
    for each in RANKED_FILE_CONTENT:
        total = total + len(each)
    for query in RANKED_FILE_CONTENT:
        queryCollection = []
        for document in query:
            count += 1
            print("----------------------------------")
            print("Processing " + str(count) + "/" + str(total))
            print("----------------------------------")
            tempDocument = document
            docid = document[1]
            qid = document[0]
            docContents = getDocumentContent(CONF, docid)
            queryContents = QUERY[qid]
            score = getPredictionScore(docContents, queryContents)
            tempDocument.append(score)
            queryCollection.append(tempDocument)
        sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
        rankedCollection.append(sortedQueryCollection)
    return rankedCollection


def generateResFile(rankedCollection, CONF):
    resPath = CONF["RESULT"]
    for query in rankedCollection:
        rank = 1
        lines = []
        for document in query:
            line = document[0] + "\t" + document[1] + "\t" + str(rank) + "\t" + str(document[3]) + "\n"
            rank += 1
            lines.append(line)
        with open(resPath + "gpt2_large_rerank.res", "a+") as f:
            f.writelines(lines)
