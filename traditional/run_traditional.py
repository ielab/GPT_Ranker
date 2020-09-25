from pyserini.search import SimpleSearcher
import json
from tqdm import tqdm


def readFile(path: str):
    with open(path, "r") as f:
        lines = f.readlines()
        res = []
        for line in lines:
            json_line = json.loads(line)
            res.append(json_line)
    return res


def run():
    query = readFile('/Volumes/IELab/ielab/GPT_Ranker/data/pass_rerank/query/pass-query-dev.small.json')
    searcher = SimpleSearcher('/Volumes/IELab/ielab/GPT_Ranker/data/pass_rerank/index/lucene-index-msmarco')
    searcher.unset_rm3()
    searcher.set_qld(2500)

    for q in tqdm(query):
        qid = q['id']
        q_content = q['contents']
        res = searcher.search(q_content, k=1000)
        with open('/Volumes/IELab/ielab/GPT_Ranker/traditional/runs/pass_dev_qlm.small.res', "a+") as f:
            for i in range(len(res)):
                f.write(f'{qid} Q0 {res[i].docid:15} {i + 1} {res[i].score:.5f} qlm')
    f.close()


if __name__ == '__main__':
    run()