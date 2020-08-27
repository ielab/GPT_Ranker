import sys

import matplotlib.pyplot as plt
import numpy as np
from trectools import TrecRes

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("not enough arguments specified")
        sys.exit(1)
    measure = sys.argv[1]
    output = sys.argv[2]
    fnames = sys.argv[3:]

    evs = []
    for fname in fnames:
        evs.append(TrecRes(fname))

    res = []
    for ev in evs:
        res.append(list(ev.get_results_for_metric(measure).values()))

    ind = np.arange(len(fnames))

    # INSERT CODE TO CREATE THE VISUALISATION HERE.
    # HINT: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html
    
    trimmed_names = []
    for name in fnames:
        name = name.rsplit("/", 1)[1]
        trimmed_names.append(name)

    plt.xticks(ind, trimmed_names, rotation="vertical")
    plt.boxplot(res, labels=trimmed_names)
    plt.ylim(0, 1)
    plt.title('Runs comparison')
    plt.ylabel(measure)
    plt.tight_layout()
    plt.show()
    # plt.savefig(output)
