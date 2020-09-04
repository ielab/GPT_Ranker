from helper import readQueryFile
import linecache

from torch.utils.data import Dataset, DataLoader


class msmarco_dataset(Dataset):
    def __init__(self, path, query_path):
        self._path = path
        with open(path, "r") as f:
            self._total_data = len(f.readlines()) - 1

        self.query_dic = readQueryFile(query_path)

    def __getitem__(self, index):
        line = linecache.getline(self._path, index + 1)
        # csv_line = csv.reader([line])
        docid, content = line.split("\t")
        return docid, content

    def __len__(self):
        return self._total_data


def collate_fn(batch):
    docids = []
    contents = []
    for docid, content in batch:
        docids.append(docid)
        contents.append(content)

    return docids, contents


dataset = msmarco_dataset("no_title_sliding_window_segment_ids_texts-1.txt",
                          "../data/doc_rerank/query/doc-msmarco-dev-queries.json")

loader = DataLoader(dataset,
                    batch_size=64,
                    shuffle=False,
                    collate_fn=collate_fn,
                    pin_memory=True)
#
for docids, contents in loader:
    print(docids)
    # break
