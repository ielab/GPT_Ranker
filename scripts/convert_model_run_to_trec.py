import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Create Normal Ranking File From Model Output and BM25 Run File")
parser.add_argument('--model_run', required=True, default='', help='')
parser.add_argument('--bm25_run', required=True, default='', help='')
parser.add_argument('--bm25_run_format', required=True, default='', help='trec, msmarco')
parser.add_argument('--output_path', required=True, default='', help='')
args = parser.parse_args()


model_run_path = args.model_run
bm25_run_path = args.bm25_run
bm25_format = args.bm25_run_format
out_path = args.output_path


with open(bm25_run_path, "r") as bm25_file, open(model_run_path, "r") as model_run_file, open(out_path, "a+") as out_file:
    bm25_lines = bm25_file.readlines()
    model_lines = model_run_file.readlines()
    for ind, line in enumerate(tqdm(bm25_lines)):
        if bm25_format == 'trec':
            qid, _, wid, _, _, _ = line.rstrip().split(" ")
        else:
            qid, wid, _, _ = line.rstrip().split("\t")
        new_line = f'{qid}\t{wid}\t{1}\t{model_lines[ind].rstrip()}\n'
        out_file.write(new_line)
    print("Done!")
