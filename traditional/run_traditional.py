from pyserini.search import SimpleSearcher


def readFile(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
        init_qid = lines[0].split('\t')[0]
        ranked_list = {}
        for line in lines:
            qid, docid, _ = line.split('\t')
            if qid != init_qid:
                ranked_list[qid] = [docid]
                continue
            else:
                ranked_list[qid].append(docid)
                continue


def run():
    searcher = SimpleSearcher('/Volumes/IELab/ielab/GPT_Ranker/data/pass_rerank/index/lucene-index-msmarco')
    searcher.set_qld(2500)

    res = searcher.search('brother', k=1000)

    for i in range(10):
        print(f'{res[i].docid:15} {res[i].score:.5f}')


if __name__ == '__main__':
    run()