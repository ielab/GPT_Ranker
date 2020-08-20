import json
from tqdm import tqdm

"""
THIS FILE MAINLY PROVIDES HELPER FUNCTIONS TO OTHER FUNCTIONS
"""


def readRankFile(CONF, SCONF):
    # Read the ranked file from local, which is the run.msmarco-passage.dev.small.tsv file
    rankFilePath = SCONF["RANK_FILE"]

    with open(rankFilePath, 'r') as f:
        file = f.readlines()

    # Clean the collection by removing the trailing \n and split by \t
    cleanedCol = []
    for each in file:
        temp = each.replace("\n", "")
        if CONF["METHOD"] == "PASS":
            temp = temp.split("\t")
            cleanedCol.append(temp)
        else:
            temp = temp.split(" ")
            newtemp = [temp[0], temp[2], temp[3]]
            cleanedCol.append(newtemp)

    fullCollection = []
    tempCollection = []
    lastCollection = []
    currentID = cleanedCol[0][0]

    # Separate the collection by different query ids, this will result in a 6980 length list
    # with each contains the document ids that it retrieved (normally 1000 per query, but some are less)
    for element in cleanedCol:
        if element[0] == currentID:
            tempCollection.append(element)
        else:
            currentID = element[0]
            fullCollection.append(tempCollection)
            tempCollection = [element]
        if element[0] == cleanedCol[-1][0]:
            lastCollection.append(element)
    fullCollection.append(lastCollection)

    return fullCollection


def readQueryFile(CONF):
    # Read the query file into memory and construct a dictionary
    queryCollection = []
    queryDict = {}
    queryFilePath = CONF["QUERY"]

    with open(queryFilePath, 'r') as f:
        contents = f.readlines()

    for line in contents:
        queryContent = json.loads(line)
        queryCollection.append(queryContent)

    for query in queryCollection:
        queryDict[query["id"]] = query["contents"]

    return queryDict


def readCollectionFile(CONF, RANKED_FILE_CONTENT):
    dic = {}
    for query in RANKED_FILE_CONTENT:
        for document in query:
            if document[1] not in dic:
                dic[document[1]] = ""
            else:
                continue

    collectionDict = getDocumentContent(CONF, dic)

    return collectionDict


def getDocumentContent(CONF, dic):
    # Scan all 9 collection_jsonl files to find the corresponding contents
    collectionPaths = CONF["COLLECTION"]
    for path in tqdm(collectionPaths, desc='Loading Collection...'):
        with open(path, "r") as f:
            lines = f.readlines()
        for line in lines:
            jsonL = json.loads(line)
            if jsonL["id"] in dic:
                dic[jsonL["id"]] = jsonL["contents"]
            else:
                continue
    return dic
