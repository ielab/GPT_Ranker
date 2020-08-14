from tqdm import tqdm
"""
THIS FILE MAINLY CONTAINS FUNCTIONS THAT COMMUNICATE BETWEEN MAIN AND RETRIEVER/WORKER
"""


def getDocumentContentFromDict(docid, COLLECTION_DICT):
    return COLLECTION_DICT[docid]


def rerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, topK, worker, workerNum):
    # Construct the sorted re-ranked list
    if topK == 0:
        rankedCollection = []
        for query in tqdm(RANKED_FILE_CONTENT, desc="Process Query With Worker " + str(workerNum)):
            queryCollection = []
            for document in tqdm(query, desc="Process Document With Worker " + str(workerNum)):
                tempDocument = document
                docid = document[1]
                qid = document[0]
                docContents = getDocumentContentFromDict(docid, COLLECTION_DICT)
                queryContents = QUERY[qid]
                score = worker.getPredictionScore(docContents, queryContents)
                tempDocument.append(score)
                queryCollection.append(tempDocument)
            sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
            rankedCollection.append(sortedQueryCollection)
        return rankedCollection
    else:
        rankedCollection = []
        for query in tqdm(RANKED_FILE_CONTENT, desc="Process Query With Worker " + str(workerNum)):
            queryCollection = []
            innerCount = 0
            for document in tqdm(query, desc="Process Document With Worker " + str(workerNum)):
                if topK > len(query):
                    topK = len(query)
                if innerCount < topK:
                    tempDocument = document
                    docid = document[1]
                    qid = document[0]
                    docContents = getDocumentContentFromDict(docid, COLLECTION_DICT)
                    queryContents = QUERY[qid]
                    score = worker.getPredictionScore(docContents, queryContents)
                    tempDocument.append(score)
                    queryCollection.append(tempDocument)
                    innerCount += 1
            sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
            rankedCollection.append(sortedQueryCollection)
        return rankedCollection


def generateResFile(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, CONF, topK, worker, workerNum):
    rankedCollection = rerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, topK, worker, workerNum)
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
