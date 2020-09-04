from tqdm import tqdm
from sentence_splitter import SentenceSplitter
import spacy
import re
import torch
# from itertools import chain
NLP = spacy.blank("en")
NLP.add_pipe(NLP.create_pipe("sentencizer"))
"""
THIS FILE MAINLY CONTAINS FUNCTIONS THAT COMMUNICATE BETWEEN MAIN AND RETRIEVER/WORKER
"""


def getDocumentContentFromDict(docid, COLLECTION_DICT):
    return COLLECTION_DICT[docid]


def getBatchDocumentContentFromDict(docids, COLLECTION_DICT):
    result = []
    for docid in docids:
        result.append(COLLECTION_DICT[docid].lower())
    return result


def getBatchSplittedDocumentSentences(docids, COLLECTION_DICT):
    result = []
    for docid in docids:
        splitter = SentenceSplitter(language='en')
        document = COLLECTION_DICT[docid]
        sentences = splitter.split(text=document)
        for sentence in sentences:
            result.append((docid, sentence))
    return result


def getNLPbatchSlidingWindows(docids, COLLECTION_DICT):
    windows = []
    ids = []
    for docid in docids:
        doc_text = COLLECTION_DICT[docid]
        doc = NLP(doc_text[:1000000])
        sentences = [sent.string.strip() for sent in doc.sents]
        for i in range(0, len(sentences), 5):
            segment = ' '.join(sentences[i:i + 10])
            segment = re.sub(r'^#*', '', segment)
            segment += ' </s>'
            windows.append(segment)
            ids.append(docid)
            if i + 10 >= len(sentences):
                break
    return windows, ids



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
        batchSize = 64

        encoded_decoder_inputs = worker.tokenize([queryContents] * batchSize)


        if topK > len(query):
            temp = len(query)
        else:
            temp = topK

        if CONF["METHOD"] == "PASS":
            queryCollection = []

            numIter = temp // batchSize + 1

            for iter in tqdm(range(numIter), desc="Process Document With Worker " + str(workerNum)):
                start = iter * batchSize
                end = (iter + 1) * batchSize
                if end > temp:
                    end = temp
                    encoded_decoder_inputs = worker.tokenize([queryContents] * (end - start))
                docids = query[start:end, 1]
                ranks = query[start:end, 2]
                batchDocContents = getBatchDocumentContentFromDict(docids, COLLECTION_DICT)
                scores = worker.batchPredict(batchDocContents, encoded_decoder_inputs, SCONF)
                for i in range(len(scores)):
                    queryCollection.append([qid, docids[i], ranks[i], scores[i]])

            sortedQueryCollection = sorted(queryCollection, key=lambda l: l[3], reverse=True)
            lines = []
            for inde, document in enumerate(sortedQueryCollection):
                line = document[0] + "\t" + document[1] + "\t" + str(inde + 1) + "\t" + str(document[3]) + "\n"
                lines.append(line)
            with open("{}{}/{}_rerank_{}-{}_lower.res".format(resPath, CONF["MODEL_NAME"], CONF["MODEL_NAME"], topK, workerNum), "a+") as f:
                f.writelines(lines)

        else:

            docids = query[:temp, 1]
            windows, ids = getNLPbatchSlidingWindows(docids, COLLECTION_DICT)
            num_sentences = len(ids)
            numIter = num_sentences // batchSize + 1

            if num_sentences % batchSize == 0:
                numIter -= 1

            temp_socres = torch.Tensor([0] * num_sentences)

            for iter in tqdm(range(numIter), desc="Process Document With Worker " + str(workerNum)):
                start = iter * batchSize
                end = (iter + 1) * batchSize
                if end > num_sentences:
                    end = num_sentences
                    encoded_decoder_inputs = worker.tokenize([queryContents] * (end-start))

                scores = worker.batchPredict(windows[start:end], encoded_decoder_inputs, SCONF)
                temp_socres[start:end] = scores[:]

            temp_socres = temp_socres.tolist()
            with open("result/doc_rerank/t5/sentence_scoring/t5_doc_sentence_scoring.txt", "a+") as f:
                for i in range(len(ids)):
                    f.write("{}\t{}\t{}\n".format(qid, ids[i], temp_socres[i]))

            # fullSentenceList = list(chain.from_iterable(documentInSentences))
            # CHUNK_SIZE = SCONF["CHUNK_SIZE"]
            #
            # chunkedSentenceList = [fullSentenceList[i:i + CHUNK_SIZE] for i in range(0, len(fullSentenceList), CHUNK_SIZE)]
            # positionList = [{} for _ in range(len(chunkedSentenceList))]
            #
            # index = 0
            #
            # for ind, sentences in enumerate(documentInSentences):
            #     total = sum(positionList[index].values())
            #     if len(sentences) + total < CHUNK_SIZE:
            #         positionList[index][docids[ind]] = len(sentences)
            #     elif len(sentences) + total == CHUNK_SIZE:
            #         positionList[index][docids[ind]] = len(sentences)
            #         index += 1
            #     else:
            #         need = CHUNK_SIZE - total
            #         rest = total + len(sentences) - CHUNK_SIZE
            #         positionList[index][docids[ind]] = need
            #         index += 1
            #         positionList[index][docids[ind]] = rest
            #
            # for ind, sentences in enumerate(chunkedSentenceList):
            #     count = 0
            #     scores = worker.batchPredict(sentences, queryContents, SCONF)
            #     innerCount = 0
            #     positionItem = positionList[ind]
            #     positionKeys = positionItem.keys()
            #     positionValues = positionItem.values()
            #     for i in range(positionValues[innerCount]):
            #         with open("result/doc_rerank/t5/sentence_scoring/t5_doc_sentence_scoring.txt", "a+") as f:
            #             f.write("{}\t{}_{}\t{}\n".format(qid, positionKeys[innerCount], i, scores[count]))
            #             count += 1
            #     innerCount += 1
