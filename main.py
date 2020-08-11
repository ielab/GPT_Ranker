import json
from helper import *
from middleware import *

print("--------------------------------")
print("Loading Config...")
f = open("config.json")
CONF = json.loads(f.read())
f.close()
print("Loaded")
print("--------------------------------")
print("Loading Ranked File...")
RANKED_FILE_CONTENT = readRankFile(CONF)
print("Loaded")
print("--------------------------------")
print("Loading Query File...")
QUERY = readQueryFile(CONF)
print("Loaded")
print("--------------------------------")


def main():
    for query in RANKED_FILE_CONTENT:
        for document in query:
            docid = document[1]
            qid = document[0]
            docContents = getDocumentContent(CONF, docid)
            queryContents = QUERY[qid]
            print(docContents)
            print(queryContents)


if __name__ == '__main__':
    main()
