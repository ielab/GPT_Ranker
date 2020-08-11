import json


def readRankFile(CONF):
    rankFilePath = CONF["RANK_FILE"]

    with open(rankFilePath, 'r') as f:
        file = f.readlines()

    cleanedCol = []
    for each in file:
        temp = each.replace("\n", "")
        temp = temp.split("\t")
        cleanedCol.append(temp)

    fullCollection = []
    tempCollection = []
    lastCollection = []
    currentID = cleanedCol[0][0]

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
