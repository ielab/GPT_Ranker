import argparse
from tqdm import tqdm


def load_file(path, form):
    print('Loading Input File...')
    file_content = {}
    if form == 'trec':
        for line in tqdm(open(path), desc='Reading...'):
            qid, _, wid, _, score, _ = line.split(' ')
            did = wid.split('_')[0]
            if qid not in file_content:
                file_content[qid] = {
                    did: [float(score)]
                }
            else:
                if did not in file_content[qid]:
                    file_content[qid][did] = [float(score)]
                else:
                    file_content[qid][did].append(float(score))
    else:
        for line in tqdm(open(path), desc='Reading...'):
            qid, wid, rank, score = line.split('\t')
            did = wid.split('_')[0]
            if qid not in file_content:
                file_content[qid] = {
                    did: [float(score)]
                }
            else:
                if did not in file_content[qid]:
                    file_content[qid][did] = [float(score)]
                else:
                    file_content[qid][did].append(float(score))
    return file_content


parser = argparse.ArgumentParser(description="Create Normal Ranking File From Sliding Window Runs")
parser.add_argument('--input_path', required=True, default='', help='')
parser.add_argument('--input_format', required=True, default='', help='trec, msmarco')
parser.add_argument('--output_path', required=True, default='', help='')
parser.add_argument('--aggregation_method', required=False, default='max', help='max, sum, occurrence')
parser.add_argument('--cutoff', required=False, default=0, help='')
args = parser.parse_args()


def run_aggregation():
    results = []
    m = args.aggregation_method
    cutoff = int(args.cutoff)

    f_content = load_file(args.input_path, args.input_format)

    if m == 'sum':
        for query, did in tqdm(f_content.items(), desc='Looping qids...'):
            cut = 0
            if cutoff > len(did.items()) or cutoff == 0:
                cutoff = len(did.items())
            if cut < cutoff:
                for docid, scores in tqdm(did.items(), desc='Looping dids...'):
                    result = [query, docid, sum(scores)]
                    results.append(result)
                cut += 1
            else:
                break
    elif m == 'max':
        for query, did in tqdm(f_content.items(), desc='Looping qids...'):
            cut = 0
            if cutoff > len(did.items()) or cutoff == 0:
                cutoff = len(did.items())
            if cut < cutoff:
                for docid, scores in tqdm(did.items(), desc='Looping dids...'):
                    result = [query, docid, max(scores)]
                    results.append(result)
                cut += 1
            else:
                break
    elif m == 'occurrence':
        for query, did in tqdm(f_content.items(), desc='Looping qids...'):
            cut = 0
            if cutoff > len(did.items()) or cutoff == 0:
                cutoff = len(did.items())
            if cut < cutoff:
                for docid, scores in tqdm(did.items(), desc='Looping dids...'):
                    result = [query, docid, len(scores)]
                    results.append(result)
                cut += 1
            else:
                break
    elif m == 'average':
        for query, did in tqdm(f_content.items(), desc='Looping qids...'):
            cut = 0
            if cutoff > len(did.items()) or cutoff == 0:
                cutoff = len(did.items())
            if cut < cutoff:
                for docid, scores in tqdm(did.items(), desc='Looping dids...'):
                    result = [query, docid, sum(scores)/len(scores)]
                    results.append(result)
                cut += 1
            else:
                break
    return sorted(results, key=lambda i: i[2], reverse=True)


res = run_aggregation()
res = sorted(res, key=lambda i: i[0], reverse=False)

print('Writing file...')
method = args.aggregation_method
out_trec_path = f'{args.output_path.rsplit(".", 1)[0]}.{method}.cutoff-{args.cutoff}.trec.res'
out_msmarco_path = f'{args.output_path.rsplit(".", 1)[0]}.{method}.cutoff-{args.cutoff}.txt'
fout_trec = open(out_trec_path, "a+")
fout_msmarco = open(out_msmarco_path, "a+")

current_qid = res[0][0]
rank = 1
for line in tqdm(res):
    qid = line[0]
    if qid == current_qid:
        fout_trec.write(f'{line[0]} Q0 {line[1]} {rank} {line[2]} sliding_{method}\n')
        fout_msmarco.write(f'{line[0]}\t{line[1]}\t{rank}\t{line[2]}\n')
        rank += 1
    else:
        current_qid = line[0]
        rank = 1
        fout_trec.write(f'{line[0]} Q0 {line[1]} {rank} {line[2]} sliding_{method}\n')
        fout_msmarco.write(f'{line[0]}\t{line[1]}\t{rank}\t{line[2]}\n')

fout_msmarco.close()
fout_trec.close()
print('Done!')
