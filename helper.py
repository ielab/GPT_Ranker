import json

"""
THIS FILE MAINLY PROVIDES HELPER FUNCTIONS TO OTHER FUNCTIONS
"""


def readRankFile(CONF):
    # Read the ranked file from local, which is the run.msmarco-passage.dev.small.tsv file
    rankFilePath = CONF["RANK_FILE"]

    with open(rankFilePath, 'r') as f:
        file = f.readlines()

    # Clean the collection by removing the trailing \n and split by \t
    cleanedCol = []
    for each in file:
        temp = each.replace("\n", "")
        temp = temp.split("\t")
        cleanedCol.append(temp)

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
