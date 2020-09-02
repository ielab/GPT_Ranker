# import argparse
import re
import spacy
from tqdm import tqdm
from multiprocessing import *
import numpy as np


def load_corpus(path):
    print('Loading corpus...')
    corpus = {}
    for line in tqdm(open(path)):
        doc_id, doc_url, doc_title, doc_text = line.split('\t')
        doc_text = doc_text.strip()
        corpus[doc_id] = (doc_title, doc_text)
    return corpus


# parser = argparse.ArgumentParser(
#     description='Create T5-formatted tsv file from MS MARCO Document Ranking '
#                 'dataset.')
# parser.add_argument('--corpus_path', required=False, default='', help='')
# parser.add_argument('--output_segment_texts_path', required=False, default='',
#                     help='')
# parser.add_argument('--output_segment_doc_ids_path', required=False, default='',
#                     help='')
# parser.add_argument('--stride', default=5, help='')
# parser.add_argument('--max_length', default=10, help='')
#
# args = parser.parse_args()
def sentence_spliter(corpus_paths, workerNum):
    nlp = spacy.blank("en")
    nlp.add_pipe(nlp.create_pipe("sentencizer"))

    for corpus_path in corpus_paths:
        corpus = load_corpus(path=corpus_path)

        n_segments = 0
        n_no_segments = 0
        with open("formatted_segments/sliding_window_segment_ids_texts-{}.txt".format(workerNum), 'a+') as fout_segment_texts, \
                open("formatted_segments/no_sliding_window_segment_ids_texts-{}.txt".format(workerNum), "a+") as fout, \
                open("formatted_segments/no_title_sliding_window_segment_ids_texts-{}.txt".format(workerNum), "a+") as fnt, \
                open("formatted_segments/no_title_no_sliding_window_segment_ids_texts-{}.txt".format(workerNum), "a+") as fnn:
            # open(args.output_segment_doc_ids_path, 'w') as fout_segment_doc_ids:

            for doc_id, (doc_title, doc_text) in tqdm(corpus.items(), total=len(corpus)):
                # doc = nlp(doc_text)
                doc = nlp(doc_text[:1000000])
                sentences = [sent.string.strip() for sent in doc.sents]
                for each in sentences:
                    each = re.sub(r'^#*', '', each)
                    fout.write("{}\t{}\n".format(doc_id, doc_title + ' ' + each))
                    fnn.write("{}\t{}\n".format(doc_id, each))
                if not sentences:
                    n_no_segments += 1
                for i in range(0, len(sentences), 5):
                    segment = ' '.join(sentences[i:i + 10])
                    tempp = segment
                    segment = doc_title + ' ' + segment

                    # Remove starting #'s as T5 skips those lines by default.
                    segment = re.sub(r'^#*', '', segment)
                    segment1 = re.sub(r'^#*', '', tempp)

                    # fout_segment_doc_ids.write(f'{doc_id}\n')
                    fout_segment_texts.write(f'{doc_id}\t{segment}\n')
                    fnt.write(f'{doc_id}\t{segment1}\n')
                    n_segments += 1
                    if i + 10 >= len(sentences):
                        break

        print(f'Wrote {n_segments} segments from {len(corpus)} docs.')
        print(f'There were {n_no_segments} docs without segments/sentences.')

        print('Done!')


if __name__ == '__main__':

    totalProcess = 8  # cpu_count()

    paths = [
        "msmarco-docs.tsv00", "msmarco-docs.tsv01", "msmarco-docs.tsv02", "msmarco-docs.tsv03", "msmarco-docs.tsv04",
        "msmarco-docs.tsv05", "msmarco-docs.tsv06", "msmarco-docs.tsv07", "msmarco-docs.tsv08", "msmarco-docs.tsv09",
        "msmarco-docs.tsv10", "msmarco-docs.tsv11", "msmarco-docs.tsv12", "msmarco-docs.tsv13", "msmarco-docs.tsv14",
        "msmarco-docs.tsv15", "msmarco-docs.tsv16", "msmarco-docs.tsv17", "msmarco-docs.tsv18", "msmarco-docs.tsv19",
        "msmarco-docs.tsv20", "msmarco-docs.tsv21", "msmarco-docs.tsv22", "msmarco-docs.tsv23", "msmarco-docs.tsv24",
        "msmarco-docs.tsv25", "msmarco-docs.tsv26", "msmarco-docs.tsv27", "msmarco-docs.tsv28", "msmarco-docs.tsv29",
        "msmarco-docs.tsv30", "msmarco-docs.tsv31", "msmarco-docs.tsv32"
    ]

    batchPaths = np.array_split(np.array(paths), totalProcess)

    processPool = []
    for index, val in enumerate(batchPaths):
        p = Process(target=sentence_spliter, args=(batchPaths[index], index + 1))
        p.start()
        processPool.append(p)
    for p in processPool:
        p.join()
