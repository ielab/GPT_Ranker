import sys

import matplotlib.pyplot as plt
import numpy as np
from trectools import TrecRes
from collections import OrderedDict

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("not enough arguments specified")
        sys.exit(1)
    fname1 = sys.argv[1]
    fname2 = sys.argv[2]
    measure = sys.argv[3]
    output = sys.argv[4]

    ev1 = TrecRes(fname1)
    ev2 = TrecRes(fname2)

    r1 = ev1.get_results_for_metric(measure)
    r2 = ev2.get_results_for_metric(measure)

    ind = np.arange(len(r1))

    # INSERT CODE TO CREATE THE VISUALISATION HERE.
    # HINT: https://docs.scipy.org/doc/numpy/reference/generated/numpy.subtract.html

    fname1 = fname1.rsplit("/", 1)[1]
    fname2 = fname2.rsplit("/", 1)[1]

    fname1 = fname1.rsplit(".", 1)[0]
    fname2 = fname2.rsplit(".", 1)[0]

    r1 = OrderedDict(sorted(r1.items(), key=lambda t: t[0]))
    r2 = OrderedDict(sorted(r2.items(), key=lambda t: t[0]))

    plt.xticks(ind, list(r1.keys()), rotation="vertical")
    plt.bar(ind, list(np.subtract(r2.values(), r1.values())))
    plt.ylim(-0.5, 0.5)
    plt.title('{} and {}'.format(fname1, fname2))
    plt.ylabel(measure)
    plt.tight_layout()
    # plt.show()
    plt.savefig(output)
