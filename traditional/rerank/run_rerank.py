from tqdm import tqdm
import requests


with open("runs/msmarco_pass_qld.dev.small.txt", "r") as qldin, open("runs/msmarco_pass_qljm.dev.small.txt", "r") as qljmin, open("msmarco_pass_qld_rerank.dev.small.txt", "a+") as qldrerank, open("msmarco_pass_qljm_rerank.dev.small.txt", "a+") as qljmrerank, open("/Volumes/IELab/ielab/GPT_Ranker/data/pass_rerank/runs/dev/run.msmarco-passage.dev.small.tsv", "r") as bm25:
    qldlines = qldin.readlines()
    qljmlines = qljmin.readlines()
    bm25lines = bm25.readlines()

    qldDict = {}
    qljmDict = {}
    bm25Dict = {}

    for qldline in tqdm(qldlines, desc='Create QLD Dict'):
        qldline_parts = qldline.strip().split(' ')
        if qldline_parts[0] not in qldDict.keys():
            qldDict[qldline_parts[0]] = {}

    for qldline in tqdm(qldlines, desc='Create QLD Dict'):
        qldline_parts = qldline.strip().split(' ')
        qldDict[qldline_parts[0]][qldline_parts[2]] = float(qldline_parts[4])

    for qljmline in tqdm(qljmlines, desc='Create QLJM Dict'):
        qljmline_parts = qljmline.strip().split(' ')
        if qljmline_parts[0] not in qljmDict.keys():
            qljmDict[qljmline_parts[0]] = {}

    for qljmline in tqdm(qljmlines, desc='Create QLJM Dict'):
        qljmline_parts = qljmline.strip().split(' ')
        qljmDict[qljmline_parts[0]][qljmline_parts[2]] = float(qljmline_parts[4])

    for bm25line in tqdm(bm25lines, desc='Create BM25 Dict'):
        bm25line_parts = bm25line.strip().split('\t')
        if bm25line_parts[0] not in bm25Dict.keys():
            bm25Dict[bm25line_parts[0]] = {}

    for bm25line in tqdm(bm25lines, desc='Create BM25 Dict'):
        bm25line_parts = bm25line.strip().split('\t')
        bm25Dict[bm25line_parts[0]][bm25line_parts[1]] = 0

    bm25qids = bm25Dict.keys()

    newQldList = []
    newQljmList = []

    for qid in tqdm(bm25qids, desc='Check Query'):
        qldlist = qldDict[qid].keys()
        qljmlist = qljmDict[qid].keys()
        bm25list = bm25Dict[qid].keys()

        for docid in tqdm(bm25list, desc='Fill New QLD List'):
            if docid in qldlist:
                newQldList.append([qid, docid, float(qldDict[qid][docid])])
            else:
                newQldList.append([qid, docid, float(0)])

        for docid in tqdm(bm25list, desc='Fill New QLJM List'):
            if docid in qljmlist:
                newQljmList.append([qid, docid, float(qljmDict[qid][docid])])
            else:
                newQljmList.append([qid, docid, float(0)])

    ranked_qld = sorted(newQldList, key=lambda x: (x[0], x[2]), reverse=True)
    ranked_qljm = sorted(newQljmList, key=lambda k: (k[0], k[2]), reverse=True)

    qldrank = 1
    qljmrank = 1
    currentqldqid = ranked_qld[0][0]
    currentqljmqid = ranked_qljm[0][0]

    for each in tqdm(ranked_qld, desc='Write QLD'):
        if each[0] == currentqldqid:
            qldrerank.write('{} Q0 {} {} {}\n'.format(each[0], each[1], qldrank, each[2], 'qld_rerank'))
            qldrank += 1
        else:
            qldrank = 1
            currentqldqid = each[0]
            qldrerank.write('{} Q0 {} {} {}\n'.format(each[0], each[1], qldrank, each[2], 'qld_rerank'))
            qldrank += 1

    for item in tqdm(ranked_qljm, desc='Write QLJM'):
        if item[0] == currentqljmqid:
            qljmrerank.write('{} Q0 {} {} {}\n'.format(item[0], item[1], qljmrank, item[2], 'qljm_rerank'))
            qljmrank += 1
        else:
            qljmrank = 1
            currentqljmqid = item[0]
            qljmrerank.write('{} Q0 {} {} {}\n'.format(item[0], item[1], qljmrank, item[2], 'qljm_rerank'))
            qljmrank += 1

qldin.close()
qljmin.close()
qldrerank.close()
qljmrerank.close()
bm25.close()
