from tqdm import tqdm
"""
THIS FILE MAINLY CONTAINS FUNCTIONS THAT COMMUNICATE BETWEEN MAIN AND RETRIEVER/WORKER
"""


def getDocumentContentFromDict(docid, COLLECTION_DICT):
    return COLLECTION_DICT[docid]


def rerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, CONF, SCONF, QUERY, topK, worker, workerNum):
    # Construct the sorted re-ranked list
    resPath = SCONF["RESULT"]
    for query in tqdm(RANKED_FILE_CONTENT, desc="Process Query With Worker " + str(workerNum)):
        queryCollection = []
        innerCount = 0
        if topK > len(query):
            topK = len(query)
        for document in tqdm(query, desc="Process Document With Worker " + str(workerNum)):
            if innerCount < topK:
                tempDocument = list(document)
                docid = document[1]
                qid = document[0]
                docContents = getDocumentContentFromDict(docid, COLLECTION_DICT)
                queryContents = QUERY[qid]
                score = worker.getPredictionScore(docContents, queryContents)
                tempDocument.append(score)
                queryCollection.append(tempDocument)
                innerCount += 1
            else:
                break
        sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
        lines = []
        for inde, document in enumerate(sortedQueryCollection):
            line = document[0] + "\t" + document[1] + "\t" + str(inde + 1) + "\t" + str(document[3]) + "\n"
            lines.append(line)
        with open("{}{}/{}_rerank-{}.res".format(resPath, CONF["MODEL_NAME"], CONF["MODEL_NAME"], workerNum), "a+") as f:
            f.writelines(lines)
