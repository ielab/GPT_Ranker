from helper import *
from middleware import *
from multiprocessing import *
from ranker import *
import numpy as np
try:
    set_start_method('spawn')
except RuntimeError:
    print("fail to set spawn")
print("--------------------------------------------------")
print("Loading Config/BM25 Retrieved List/Query List/Collection Dict...")
f = open("config.json")
CONF = json.loads(f.read())
f.close()
if CONF["METHOD"] == "PASS":
    SCONF = CONF["PASS_CONF"]
    RANKED_FILE_CONTENT = readRankFile(CONF, SCONF)
    QUERY = readQueryFile(SCONF)
    COLLECTION_DICT = readCollectionFile(SCONF, RANKED_FILE_CONTENT)
    if CONF["MODEL_NAME"] == "gpt2":
        WORKER = GPT2()
    else:
        WORKER = T5(CONF["T5_MODEL"])
    print("Loaded")
else:
    SCONF = CONF["DOC_CONF"]
    RANKED_FILE_CONTENT = readRankFile(CONF, SCONF)
    QUERY = readQueryFile(SCONF)
    COLLECTION_DICT = readCollectionFile(SCONF, RANKED_FILE_CONTENT)
    if CONF["MODEL_NAME"] == "gpt2":
        WORKER = GPT2()
    else:
        WORKER = T5(CONF["T5_MODEL"])
    print("Loaded")
print("--------------------------------------------------")


def getResFiles():
    topK = 1000
    rerankDocuments(RANKED_FILE_CONTENT, COLLECTION_DICT, CONF, SCONF, QUERY, topK, WORKER, 1)


if __name__ == '__main__':
    totalProcess = 8  # cpu_count()

    chunkRes = np.array_split(np.array(RANKED_FILE_CONTENT), totalProcess)
    processPool = []
    for index, val in enumerate(chunkRes):
        p = Process(target=batchRerankDocuments, args=(chunkRes[index], COLLECTION_DICT, CONF, SCONF, QUERY, 200, WORKER, index + 1))
        p.start()
        processPool.append(p)
    for p in processPool:
        p.join()
    print("Finished")
