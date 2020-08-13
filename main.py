from helper import *
from middleware import *
from multiprocessing import *

print("--------------------------------------------------")
print("Loading Config/BM25 Retrieved List/Query List/Collection Dict...")
f = open("config.json")
CONF = json.loads(f.read())
f.close()
RANKED_FILE_CONTENT = readRankFile(CONF)
QUERY = readQueryFile(CONF)
COLLECTION_DICT = readCollectionFile(CONF,RANKED_FILE_CONTENT)
print("Loaded")
print("--------------------------------------------------")


def getResFiles():
    topK = 100
    generateResFile(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, CONF, topK)


if __name__ == '__main__':
    totalProcess = cpu_count()
    chunkRes = numpy.array_split(numpy.array(RANKED_FILE_CONTENT), totalProcess)
    processPool = []
    for index, val in enumerate(chunkRes):
        p = Process(target=generateResFile, args=(chunkRes[index], COLLECTION_DICT, QUERY, CONF, 0))
        p.start()
        processPool.append(p)
    for p in processPool:
        p.join()
    print("Finished")
