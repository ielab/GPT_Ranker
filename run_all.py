import sys

import matplotlib.pyplot as plt
import numpy as np
from trectools import TrecRes
from collections import OrderedDict

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("not enough arguments specified")
        sys.exit(1)
    measure = sys.argv[1]
    output = sys.argv[2]
    fnames = sys.argv[3:]

if len(fnames) == 1:
  ev = TrecRes(fnames[0])

  r = ev.get_results_for_metric(measure)

  fname = fnames[0].rsplit("/", 1)[1]

  ind = np.arange(len(r))
  plt.bar(ind, list(r.values()))
  plt.xticks(ind, list(r.keys()), rotation="vertical")
  plt.ylim(0, 1)
  plt.title(fname)
  plt.ylabel(measure)
  plt.tight_layout()
  plt.show()
  # plt.savefig(output)

elif len(fnames) == 2:
  ev1 = TrecRes(fnames[0])
  ev2 = TrecRes(fnames[1])

  r1 = ev1.get_results_for_metric(measure)
  r2 = ev2.get_results_for_metric(measure)

  ind = np.arange(len(r1))
  width = 0.45

  fname1 = fnames[0].rsplit("/", 1)[1]
  fname2 = fnames[1].rsplit("/", 1)[1]

  r1 = OrderedDict(sorted(r1.items(), key=lambda t: t[0]))
  r2 = OrderedDict(sorted(r2.items(), key=lambda t: t[0]))

  plt.xticks(ind, list(r1.keys()), rotation="vertical")
  plt.bar(ind - width/2, list(r1.values()), width, label=fname1)
  plt.bar(ind + width/2, list(r2.values()), width, label=fname2)
  plt.ylim(0, 1)
  plt.title('{} and {}'.format(fname1, fname2))
  plt.ylabel(measure)
  plt.legend()
  plt.tight_layout()
  plt.show()
  # plt.savefig(output)

  plt.xticks(ind, list(r1.keys()), rotation="vertical")
  plt.bar(ind, list(np.subtract(r2.values(), r1.values())))
  plt.ylim(-1, 1)
  plt.title('{} and {}'.format(fname1, fname2))
  plt.ylabel(measure)
  plt.tight_layout()
  plt.show()
  # plt.savefig(output)

elif len(fnames) > 2:
  evs = []
  for fname in fnames:
      evs.append(TrecRes(fname))

  res = []
  for ev in evs:
      res.append(list(ev.get_results_for_metric(measure).values()))

  ind = np.arange(len(fnames))
  
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