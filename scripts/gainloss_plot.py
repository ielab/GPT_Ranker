import sys

import matplotlib.pyplot as plt
import numpy as np
from trectools import TrecRes
from collections import OrderedDict

if __name__ == "__main__":
    # if len(sys.argv) < 5:
    #     print("not enough arguments specified")
    #     sys.exit(1)
    # fname1 = sys.argv[1]
    # fname2 = sys.argv[2]
    # measure = sys.argv[3]
    # output = sys.argv[4]

    files = [
                [
                    "/Users/hangli/ielab/GPT_Ranker/images/bm25_T5QLM-ndcg.10.eval",
                    "/Users/hangli/ielab/GPT_Ranker/images/bm25-ndcg.10.eval",
                    "BM25+QLM-T5 and BM25"
                ],
                [
                    "/Users/hangli/ielab/GPT_Ranker/images/qld_t5-ndcg.10.eval",
                    "/Users/hangli/ielab/GPT_Ranker/images/qld-ndcg.10.eval",
                    "QLM-D+QLM-T5 and QLM-D"
                ],
                [
                    "/Users/hangli/ielab/GPT_Ranker/images/qljm_t5-ndcg.10.eval",
                    "/Users/hangli/ielab/GPT_Ranker/images/qljm-ndcg.10.eval",
                    "QLM-JM+QLM-T5 and QLM-JM"
                ]
            ]

    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(18, 4.5))

    for set_num, paths in enumerate(files):

        ev1 = TrecRes(paths[0])
        ev2 = TrecRes(paths[1])

        r1 = ev1.get_results_for_metric('ndcg_cut_10')
        r2 = ev2.get_results_for_metric('ndcg_cut_10')

        ind = np.arange(len(r1))

        # INSERT CODE TO CREATE THE VISUALISATION HERE.
        # HINT: https://docs.scipy.org/doc/numpy/reference/generated/numpy.subtract.html

        r1 = OrderedDict(sorted(r1.items(), key=lambda t: t[0]))
        r2 = OrderedDict(sorted(r2.items(), key=lambda t: t[0]))

        gains = sorted(list(np.subtract(list(r1.values()), list(r2.values()))), reverse=True)

        if set_num == 0:
            axes[set_num].set_ylabel("nDCG@10 Gain", fontsize=17)
        axes[set_num].bar(ind, gains)
        axes[set_num].set_ylim([-1.1, 1.1])
        axes[set_num].set_title(paths[2], fontsize=17)
        axes[set_num].tick_params(axis='y', labelsize=14)
        axes[set_num].tick_params(axis='x', labelsize=14)
    plt.subplots_adjust(left=0.07, bottom=0.1, right=0.98, top=0.9, wspace=0.06)
    plt.savefig('/Users/hangli/ielab/GPT_Ranker/images/gain_comparison.png')
    # plt.show()
