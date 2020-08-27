import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict

# Usage:
# python3 rank_comparison.py topK baseline.res better.res relevant.qrels

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("not enough arguments specified")
        sys.exit(1)
    topK = int(sys.argv[1])
    fname1 = sys.argv[2]
    fname2 = sys.argv[3]
    fname3 = sys.argv[4]

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

    with open(fname1, "r") as f1:
        lines1 = f1.readlines()
        for linee in tqdm(lines1, desc="Constructing Dictionary 1..."):
            segs = linee.rstrip().split(" ")
            if segs[0] not in dic1 and segs[2] == dic3[segs[0]]:
                dic1[segs[0]] = {
                    segs[2]: segs[3]
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
                    s[2]: s[3]
                }
            else:
                continue
    f2.close()

    topics = dic3.keys()

    gainDict = {}

    for topic in tqdm(topics, desc="Calculating Gain..."):
        if topic in dic1 and topic in dic2:
            gain = int(list(dic2[topic].values())[0]) - int(list(dic1[topic].values())[0])
        elif topic in dic1 and topic not in dic2:
            if int(list(dic1[topic].values())[0]) > topK:
                gain = 0
            else:
                gain = -int(list(dic1[topic].values())[0])
        elif topic in dic2 and topic not in dic1:
            if int(list(dic2[topic].values())[0]) > topK:
                gain = 0
            else:
                gain = int(list(dic2[topic].values())[0])
        else:
            gain = 0
        gainDict[topic] = gain

    orderedDict = OrderedDict(sorted(gainDict.items(), key=lambda t: t[1], reverse=True))

    ind = np.arange(len(topics))

    fname1 = fname1.rsplit("/", 1)[1]
    fname2 = fname2.rsplit("/", 1)[1]

    fname1 = fname1.rsplit(".", 1)[0]
    fname2 = fname2.rsplit(".", 1)[0]

    # plt.xticks(ind, list(gainDict.keys()), rotation="vertical")
    plt.bar(ind, list(orderedDict.values()))
    plt.ylim(-topK, topK)
    plt.title('{} Beats {} By How Much'.format(fname2, fname1))
    plt.ylabel("Ranked Up")
    plt.tight_layout()
    plt.show()
