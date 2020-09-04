from helper import readQueryFile
import linecache
from torch.utils.data import Dataset, DataLoader
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import numpy as np
from tqdm import tqdm

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class msmarco_dataset(Dataset):
    def __init__(self, path, query_path):
        self._path = path
        with open(path, "r") as f:
            self._total_data = len(f.readlines())

        self._query_dic = readQueryFile(query_path)

    def __getitem__(self, index):
        line = linecache.getline(self._path, index+1)
        qid, docid, content = line.split("\t")
        query = self._query_dic[qid]
        return query, content

    def __len__(self):
        return self._total_data


def collate_fn(batch):
    querys = []
    contents = []
    for query, content in batch:
        querys.append(query)
        contents.append(content)

    return querys, contents


def main():
    data_path = "msmarco_dev_run_collection.txt"
    query_path = "../data/doc_rerank/query/doc-msmarco-dev-queries.json"
    model_dir = "../model/t5-base/model.ckpt-1004000"

    # Loading model and put it in GPU
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    config = T5Config.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir, from_tf=True, config=config)
    model.eval()
    model.to(DEVICE)

    # softmax function can be used to compute scores on GPU
    softmax = torch.nn.Softmax(dim=1).to(DEVICE)

    # create torch DataLoader.
    dataset = msmarco_dataset(data_path,
                              query_path)
    loader = DataLoader(dataset,
                        batch_size=4,
                        shuffle=False,
                        collate_fn=collate_fn,
                        pin_memory=True)

    # for each batch
    for querys, contents in tqdm(loader):

        # batch encoding
        encoder_inputs = tokenizer(contents, padding=True, truncation=True, return_tensors="pt").to(
            DEVICE)
        decoder_inputs = tokenizer(querys, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        decoder_input_ids = decoder_inputs["input_ids"]
        # inference logits
        scores = []
        with torch.no_grad():
            outputs = model(input_ids=encoder_inputs["input_ids"],
                            labels=decoder_inputs["input_ids"],
                            attention_mask=encoder_inputs["attention_mask"],
                            decoder_attention_mask=decoder_inputs["attention_mask"])
        batch_logits = outputs[1]
        for logits in batch_logits:
            distributions = softmax(logits)
            prob = []
            for index, val in enumerate(decoder_input_ids):
                prob.append(distributions[index][val])
            score = np.sum(np.log10(prob))
            scores.append(score)


if __name__ == '__main__':
    main()
