import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# Usage:
# python3 rank_comparison.py topK baseline.res better.res relevant.qrels

if __name__ == '__main__':
    # if len(sys.argv) != 5:
    #     print("not enough arguments specified")
    #     sys.exit(1)
    # topK = int(sys.argv[1])
    # fname1 = sys.argv[2]
    # fname2 = sys.argv[3]
    # fname3 = sys.argv[4]
    
    files = [
                [
                    "/Users/hangli/ielab/GPT_Ranker/traditional/rerank/t5_rerank_BM25_1000-1_lower.trec.res",
                    "/Users/hangli/ielab/GPT_Ranker/data/pass_rerank/runs/dev/run.msmarco-passage.dev.small.trec.res",
                    "/Users/hangli/ielab/GPT_Ranker/data/pass_rerank/qrels/pass-qrels.msmarco-dev.small.tsv",
                    "BM25+QLM-T5 and BM25"
                ],
                [
                    "/Users/hangli/ielab/GPT_Ranker/traditional/rerank/t5_rerank_Dirichlet_1000-1_lower.trec.res",
                    "/Users/hangli/ielab/GPT_Ranker/traditional/first_stage/runs/pass-dev-qld.small.txt",
                    "/Users/hangli/ielab/GPT_Ranker/data/pass_rerank/qrels/pass-qrels.msmarco-dev.small.tsv",
                    "QLM-D+QLM-T5 and QLM-D"
                ],
                [
                    "/Users/hangli/ielab/GPT_Ranker/traditional/rerank/t5_rerank_JelinekMercer_1000-1_lower.trec.res",
                    "/Users/hangli/ielab/GPT_Ranker/traditional/first_stage/runs/pass-dev-qljm.small.txt",
                    "/Users/hangli/ielab/GPT_Ranker/data/pass_rerank/qrels/pass-qrels.msmarco-dev.small.tsv",
                    "QLM-JM+QLM-T5 and QLM-JM"
                ]
            ]

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(18, 4.5))

    for set_num, paths in enumerate(files):

        fname1 = paths[0]
        fname2 = paths[1]
        fname3 = paths[2]

        dic1 = {}
        dic2 = {}
        dic3 = {}

        with open(fname3, "r") as f3:
            lines3 = f3.readlines()
            for line3 in lines3:
                ss = line3.rstrip().split("\t")
                if ss[0] not in dic3:
                    dic3[ss[0]] = ss[2]
                else:
                    continue
        f3.close()

        with open(fname1, "r") as f1:
            lines1 = f1.readlines()
            for linee in tqdm(lines1, desc="Constructing Dictionary 1..."):
                segs = linee.rstrip().split(" ")
                if segs[0] not in dic1 and segs[2] == dic3[segs[0]]:
                    dic1[segs[0]] = {
                        segs[2]: int(segs[3]) - 1
                    }
                else:
                    continue
        f1.close()

        with open(fname2, "r") as f2:
            lines2 = f2.readlines()
            for linne in tqdm(lines2, desc="Constructing Dictionary 2..."):
                s = linne.rstrip().split(" ")
                if s[0] not in dic2 and s[2] == dic3[s[0]]:
                    dic2[s[0]] = {
                        s[2]: int(s[3]) - 1
                    }
                else:
                    continue
        f2.close()

        topics = dic3.keys()

        gainDict = {}

        for topic in tqdm(topics, desc="Calculating Gain..."):
            if topic in dic1 and topic in dic2:
                gain = list(dic1[topic].values())[0] - list(dic2[topic].values())[0]
                gainDict[topic] = -gain

        orderedDict = OrderedDict(sorted(gainDict.items(), key=lambda t: t[1], reverse=True))

        ind = np.arange(len(list(orderedDict.values())))

        if set_num == 0:
            axes[set_num].set_ylabel("Rank Gain", fontsize=17)
        axes[set_num].bar(ind, list(orderedDict.values()))
        axes[set_num].set_ylim([-1000, 1000])
        axes[set_num].set_title(paths[3], fontsize=17)
        axes[set_num].tick_params(axis='y', labelsize=14)
        axes[set_num].tick_params(axis='x', labelsize=14)
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.98, top=0.9, wspace=0.06)
    plt.savefig('/Users/hangli/ielab/GPT_Ranker/images/comparison.png')
    # plt.show()
