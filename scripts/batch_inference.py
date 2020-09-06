import linecache
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration
import json
from tqdm import tqdm
import os

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('t5-base')


def readQueryFile(path):
    # Read the query file into memory and construct a dictionary
    queryDict = {}
    queryFilePath = path

    with open(queryFilePath, 'r') as f:
        contents = f.readlines()

    for line in tqdm(contents, desc='Loading Queries...'):
        query = json.loads(line)
        queryDict[query["id"]] = query["contents"]

    return queryDict


def readDocumentFile(path):
    docDict = {}
    docFilePaths = os.listdir(path)

    for docFile in tqdm(docFilePaths, desc='Loading Documents...'):

        with open(path+'/'+docFile, 'r') as f:
            contents = f.readlines()

        for line in contents:
            document = json.loads(line)
            docDict[document["id"]] = document["contents"]

    return docDict


class msmarco_dataset(Dataset):
    def __init__(self, res_path, query_path, doc_path):
        self._res_path = res_path
        with open(self._res_path, "r") as f:
            self._total_data = len(f.readlines())
        print("total number of passages need to be rerank:",self._total_data)

        self._query_dic = readQueryFile(query_path)
        self._doc_dic = readDocumentFile(doc_path)


    def __getitem__(self, index):
        line = linecache.getline(self._res_path, index+1)
        qid, docid, _, _ = line.rstrip().split("\t")
        query = self._query_dic[qid]
        content = self._doc_dic[docid] + ' </s>'

        return query, content

    def __len__(self):
        return self._total_data


def collate_fn(batch):

    querys = []
    contents = []
    for query, content in batch:
        querys.append(query)
        contents.append(content)

    encoder_inputs = tokenizer(contents, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    decoder_inputs = tokenizer(querys, padding=True, truncation=True, return_tensors="pt").to(DEVICE)

    return encoder_inputs, decoder_inputs


def main():
    res_path = "/media/bigdata/uqhli31/GPT_Ranker/sliding_window_runs/no_title_sliding_windows/dev/doc-tuned/formatted_run.msmarco-doc.dev.bm25.doc-tuned.no-title.sliding-window-top2000.txt"
    doc_path = "../data/doc_rerank/collection_jsonl/"
    query_path = "../data/doc_rerank/query/doc-msmarco-dev-queries.json"
    model_dir = "../model/t5-base/model.ckpt-1004000"

    # Loading model and put it in GPU
    print("Loading model......")
    config = T5Config.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained(
        model_dir, from_tf=True, config=config)
    model.eval()
    model.to(DEVICE)

    # softmax function can be used to compute scores on GPU
    softmax = torch.nn.Softmax(dim=-1).to(DEVICE)

    # create torch DataLoader.
    print("Creating DataLoader......")
    dataset = msmarco_dataset(res_path, query_path, doc_path)
    loader = DataLoader(dataset,
                        batch_size=32,
                        shuffle=False,
                        collate_fn=collate_fn,)

    sf = open("t5_scores.txt", 'a+')
    batch_num = 0
    temp_scores = []
    # for each batch
    for encoder_inputs, decoder_inputs in tqdm(loader):
        batch_num += 1
        decoder_input_ids = decoder_inputs["input_ids"]  # shape(batch_size, decoder_dim)
        decoder_attention_mask = decoder_inputs["attention_mask"]  # shape(batch_size, decoder_dim)

        # inference logits
        with torch.no_grad():
            outputs = model(input_ids=encoder_inputs["input_ids"],
                            labels=decoder_input_ids,
                            attention_mask=encoder_inputs["attention_mask"],
                            decoder_attention_mask=decoder_attention_mask)
            batch_logits = outputs[1]  # shape(batch_size, decoder_dim, num_tokens)
            distributions = softmax(batch_logits)  # shape(batch_size, decoder_dim, num_tokens)
            decoder_input_ids = decoder_input_ids.unsqueeze(-1)  # shape(batch_size, decoder_dim, 1)
            batch_probs = torch.gather(distributions, 2, decoder_input_ids).squeeze(-1)  # shape(batch_size, decoder_dim)
            masked_log_probs = torch.log10(batch_probs) * decoder_attention_mask  # shape(batch_size, decoder_dim)
            socres = torch.sum(masked_log_probs, 1)  # shape(batch_size)

        temp_scores.extend(socres.tolist())
        if batch_num % 100 == 0:
            for score in temp_scores:
                sf.write(str(score)+'\n')
                temp_scores = []
    
    for score in temp_scores:
        sf.write(score+'\n')

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
    sf.close()
if __name__ == '__main__':
    main()
