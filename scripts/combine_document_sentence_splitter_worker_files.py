from tqdm import tqdm
from multiprocessing import Process

set1 = ["GPT_Ranker/docTTTTTquery/formatted_segments/title_sliding_windows/sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.dev.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/title_sliding_windows/msmarco_dev_run_collection_t_s.txt"]

set2 = ["GPT_Ranker/docTTTTTquery/formatted_segments/title_sliding_windows/sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.eval.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/title_sliding_windows/msmarco_eval_run_collection_t_s.txt"]

set3 = ["GPT_Ranker/docTTTTTquery/formatted_segments/title_no_sliding_windows/no_sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.dev.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/title_no_sliding_windows/msmarco_dev_run_collection_t_no_s.txt"]

set4 = ["GPT_Ranker/docTTTTTquery/formatted_segments/title_no_sliding_windows/no_sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.eval.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/title_no_sliding_windows/msmarco_eval_run_collection_t_no_s.txt"]

set5 = ["GPT_Ranker/docTTTTTquery/formatted_segments/no_title_sliding_windows/no_title_sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.dev.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/no_title_sliding_windows/msmarco_dev_run_collection_no_t_s.txt"]

set6 = ["GPT_Ranker/docTTTTTquery/formatted_segments/no_title_sliding_windows/no_title_sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.eval.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/no_title_sliding_windows/msmarco_eval_run_collection_no_t_s.txt"]

set7 = ["GPT_Ranker/docTTTTTquery/formatted_segments/no_title_no_sliding_windows/no_title_no_sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.dev.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/no_title_no_sliding_windows/msmarco_dev_run_collection_no_t_no_s.txt"]

set8 = ["GPT_Ranker/docTTTTTquery/formatted_segments/no_title_no_sliding_windows/no_title_no_sliding_window_segment_ids_texts.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/formatted_run.msmarco-doc.eval.bm25.tuned.txt",
        "GPT_Ranker/docTTTTTquery/file_processing/no_title_no_sliding_windows/msmarco_eval_run_collection_no_t_no_s.txt"]

sets = [set1, set2, set3, set4, set5, set6, set7, set8]


def runW(path_set, workerNum):
    with open(path_set[0], "r") as f, \
            open(path_set[1], "r") as ff, \
            open(path_set[2], "a+") as fff:
        runDict = {}
        for run in tqdm(ff, desc="{} Construct dict".format(workerNum)):
            parts = run.rstrip().split("\t")
            qid = parts[0]
            did = parts[1]
            if did in runDict:
                runDict[did].append(qid)
            else:
                runDict[did] = [qid]
        for line in tqdm(f, desc="{} Loop collection".format(workerNum)):
            segs = line.rstrip().split("\t")
            docid = segs[0]
            docContents = segs[1]
            if docid in runDict:
                for q in runDict[docid]:
                    fff.write("{}\t{}\t{}\n".format(q, docid, docContents))
            else:
                continue


if __name__ == '__main__':

    processPool = []

    for ind, each in enumerate(sets):
        p = Process(target=runW, args=(each, ind))
        p.start()
        processPool.append(p)
    for p in processPool:
        p.join()
