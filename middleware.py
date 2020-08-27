from tqdm import tqdm
"""
THIS FILE MAINLY CONTAINS FUNCTIONS THAT COMMUNICATE BETWEEN MAIN AND RETRIEVER/WORKER
"""


def getDocumentContentFromDict(docid, COLLECTION_DICT):
    return COLLECTION_DICT[docid]


def getBatchDocumentContentFromDict(docids, COLLECTION_DICT):
    result = []
    for docid in docids:
        result.append(COLLECTION_DICT[docid])
    return result


def rerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, CONF, SCONF, QUERY, topK, worker, workerNum):
    # Construct the sorted re-ranked list
    resPath = SCONF["RESULT"]
    for query in tqdm(RANKED_FILE_CONTENT, desc="Process Query With Worker " + str(workerNum)):
        queryCollection = []
        if topK > len(query):
            temp = len(query)
        else:
            temp = topK
        for i in tqdm(range(temp), desc="Process Document With Worker " + str(workerNum)):
            document = query[i]
            docid = document[1]
            qid = document[0]
            rank = document[2]
            docContents = getDocumentContentFromDict(docid, COLLECTION_DICT)
            queryContents = QUERY[qid]
            score = worker.predict(docContents, queryContents, SCONF)
            queryCollection.append([qid, docid, rank, score])

        sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
        lines = []
        for inde, document in enumerate(sortedQueryCollection):
            line = document[0] + "\t" + document[1] + "\t" + str(inde + 1) + "\t" + str(document[3]) + "\n"
            lines.append(line)
        with open("{}{}/{}_rerank_{}-{}.res".format(resPath, CONF["MODEL_NAME"], CONF["MODEL_NAME"], topK, workerNum), "a+") as f:
            f.writelines(lines)


def batchRerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, CONF, SCONF, QUERY, topK, worker, workerNum):
    # Construct the sorted re-ranked list
    resPath = SCONF["RESULT"]
    for query in tqdm(RANKED_FILE_CONTENT, desc="Process Query With Worker " + str(workerNum)):
        qid = query[0][0]
        queryContents = QUERY[qid]
        queryCollection = []
        if topK > len(query):
            temp = len(query)
        else:
            temp = topK

        batchSize = 64
        numIter = temp // 64 + 1

        for iter in tqdm(range(numIter), desc="Process Document With Worker " + str(workerNum)):
            start = iter * batchSize
            end = (iter + 1) * batchSize
            if end > temp:
                end = temp
            docids = query[start:end, 1]
            ranks = query[start:end, 2]
            batchDocContents = getBatchDocumentContentFromDict(docids, COLLECTION_DICT)
            scores = worker.batchPredict(batchDocContents, queryContents, SCONF)
            for i in range(len(scores)):
                queryCollection.append([qid, docids[i], ranks[i], scores[i]])

        sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
        lines = []
        for inde, document in enumerate(sortedQueryCollection):
            line = document[0] + "\t" + document[1] + "\t" + str(inde + 1) + "\t" + str(document[3]) + "\n"
            lines.append(line)
        with open("{}{}/{}_rerank_{}-{}.res".format(resPath, CONF["MODEL_NAME"], CONF["MODEL_NAME"], topK, workerNum), "a+") as f:
            f.writelines(lines)
