from helper import *
from middleware import *

print("--------------------------------------------------")
print("Loading Config/BM25 Retrieved List/Query List...")
f = open("config.json")
CONF = json.loads(f.read())
f.close()
RANKED_FILE_CONTENT = readRankFile(CONF)
QUERY = readQueryFile(CONF)
print("Loaded")
print("--------------------------------------------------")


def main():
    rankedCollection = rerankDocuments(RANKED_FILE_CONTENT, QUERY, CONF)
    generateResFile(rankedCollection, CONF)


if __name__ == '__main__':
    main()
