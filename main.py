from helper import *
from middleware import *
from multiprocessing import *
from gpt_worker import *

print("--------------------------------------------------")
print("Loading Config/BM25 Retrieved List/Query List/Collection Dict...")
f = open("config.json")
CONF = json.loads(f.read())
f.close()
RANKED_FILE_CONTENT = readRankFile(CONF)
QUERY = readQueryFile(CONF)
COLLECTION_DICT = readCollectionFile(CONF,RANKED_FILE_CONTENT)
WORKER = GPT2()
print("Loaded")
print("--------------------------------------------------")


def getResFiles():
    topK = 100
    generateResFile(RANKED_FILE_CONTENT, COLLECTION_DICT, QUERY, CONF, topK, WORKER)


if __name__ == '__main__':
    totalProcess = 8 # cpu_count()
    chunkRes = numpy.array_split(numpy.array(RANKED_FILE_CONTENT), totalProcess)
    processPool = []
    for index, val in enumerate(chunkRes):
        p = Process(target=generateResFile, args=(chunkRes[index], COLLECTION_DICT, QUERY, CONF, 0, WORKER))
        p.start()
        processPool.append(p)
    for p in processPool:
        p.join()
    print("Finished")
