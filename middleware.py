from gpt_worker import *
import numpy

"""
THIS FILE MAINLY CONTAINS FUNCTIONS THAT COMMUNICATE BETWEEN MAIN AND RETRIEVER/WORKER
"""


def getDocumentContentFromDict(docid, COLLECTION_DICT):
    return COLLECTION_DICT[docid]


def getPredictionScore(document: str, query: str):
    # Get the prediction score from the GPT-2 model
    worker = GPT2()
    queryTokens = query.split(" ")

    prob = worker.prediction(document, queryTokens, query)
    score = numpy.sum(numpy.log(prob))
    return score


def rerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, CONF, topK):
    # Construct the sorted re-ranked list
    if topK == 0:
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
                docContents = getDocumentContentFromDict(docid, COLLECTION_DICT)
                queryContents = QUERY[qid]
                score = getPredictionScore(docContents, queryContents)
                tempDocument.append(score)
                queryCollection.append(tempDocument)
            sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
            rankedCollection.append(sortedQueryCollection)
        return rankedCollection
    else:
        count = 0
        rankedCollection = []
        total = 0
        for each in RANKED_FILE_CONTENT:
            if len(each) > topK:
                total = total + topK
            else:
                total = total + len(each)
        for query in RANKED_FILE_CONTENT:
            queryCollection = []
            innerCount = 0
            for document in query:
                if topK > len(query):
                    topK = len(query)
                if innerCount < topK:
                    count += 1
                    print("----------------------------------")
                    print("Processing " + str(count) + "/" + str(total))
                    print("----------------------------------")
                    tempDocument = document
                    docid = document[1]
                    qid = document[0]
                    docContents = getDocumentContentFromDict(docid, COLLECTION_DICT)
                    queryContents = QUERY[qid]
                    score = getPredictionScore(docContents, queryContents)
                    tempDocument.append(score)
                    queryCollection.append(tempDocument)
                    innerCount += 1
            sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
            rankedCollection.append(sortedQueryCollection)
        return rankedCollection


def generateResFile(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, CONF, topK):
    rankedCollection = rerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, CONF, topK)
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
