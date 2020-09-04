import linecache
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import json
from tqdm import tqdm
import numpy as np

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



class msmarco_dataset(Dataset):
    def __init__(self, path, query_path):
        self._path = path
        with open(self._path, "r") as f:
            self._total_data = len(f.readlines())

        self._query_dic = readQueryFile(query_path)


    def __getitem__(self, index):
        line = linecache.getline(self._path, index+1)
        qid, docid, content = line.rstrip().split("\t")
        query = self._query_dic[qid]
        return query, content + ' </s>'

    def __len__(self):
        return self._total_data


def collate_fn(batch):
    querys = []
    contents = []
    for query, content in batch:
        querys.append(query)
        contents.append(content)

    return querys, contents


def readQueryFile(path):
    # Read the query file into memory and construct a dictionary
    queryCollection = []
    queryDict = {}
    queryFilePath = path

    with open(queryFilePath, 'r') as f:
        contents = f.readlines()

    for line in contents:
        queryContent = json.loads(line)
        queryCollection.append(queryContent)

    for query in queryCollection:
        queryDict[query["id"]] = query["contents"]

    return queryDict


def main():
    data_path = "no_title_sliding_window_segment_ids_texts-1.txt"
    query_path = "../data/doc_rerank/query/doc-msmarco-dev-queries.json"
    model_dir = "../model/t5-base/model.ckpt-1004000"

    # Loading model and put it in GPU
    print("Loading model......")
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    config = T5Config.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir, from_tf=True, config=config)
    model.eval()
    model.to(DEVICE)

    # softmax function can be used to compute scores on GPU
    softmax = torch.nn.Softmax(dim=-1).to(DEVICE)

    # create torch DataLoader.
    print("Creating DataLoader......")
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
        decoder_input_ids = decoder_inputs["input_ids"]  # shape(batch_size, decoder_dim)
        decoder_attention_mask = decoder_inputs["attention_mask"]  # shape(batch_size, decoder_dim)

        # inference logits
        with torch.no_grad():
            outputs = model(input_ids=encoder_inputs["input_ids"],
                            labels=decoder_input_ids,
                            attention_mask=encoder_inputs["attention_mask"])
            decoder_input_ids = decoder_input_ids[0]
            batch_logits = outputs[1]  # shape(batch_size, decoder_dim, num_tokens)
            distributions = softmax(batch_logits)  # shape(batch_size, decoder_dim, num_tokens)
            decoder_input_ids = decoder_input_ids.unsqueeze(-1)  # shape(batch_size, decoder_dim, 1)
            batch_probs = torch.gather(distributions, 2, decoder_input_ids).squeeze(-1)  # shape(batch_size, decoder_dim)
            # masked_log_probs = torch.log10(batch_probs) * decoder_attention_mask  # shape(batch_size, decoder_dim)
            socres = torch.sum(torch.log10(batch_probs), 1)  # shape(batch_size)
            print(socres.tolist())

            # scores = []
            # for logits in batch_logits:
            #     # distributions = softmax(logits.numpy(), axis=1)
            #     distributions = softmax(logits)
            #     prob = []
            #     for index, val in enumerate(decoder_input_ids):
            #         prob.append(distributions[index][val])
            #     score = np.sum(np.log10(prob))
            #     scores.append(score)
            #

if __name__ == '__main__':
    main()
