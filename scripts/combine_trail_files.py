from tqdm import tqdm
from multiprocessing import Process


def run(k, workerNum):
    if k == 1:
        for i in range(1, 11):
            inpath = "GPT_Ranker/docTTTTTquery/formatted_segments/no_title_sliding_windows_with_trail/no_title_sliding_window_segment_ids_texts_with_trail-{}.txt".format(
                i)
            outpath = "GPT_Ranker/docTTTTTquery/formatted_segments/no_title_sliding_windows_with_trail/no_title_sliding_window_segment_ids_texts_with_trail.txt"

            with open(inpath, "r") as fin, open(outpath, "a+") as fout:
                for line in tqdm(fin, desc="Worker {}".format(workerNum)):
                    fout.write(line)
    else:
        for i in range(1, 11):
            inpath = "GPT_Ranker/docTTTTTquery/formatted_segments/title_sliding_windows_with_trail/title_sliding_window_segment_ids_texts_with_trail-{}.txt".format(
                i)
            outpath = "GPT_Ranker/docTTTTTquery/formatted_segments/title_sliding_windows_with_trail/title_sliding_window_segment_ids_texts_with_trail.txt"

            with open(inpath, "r") as fin, open(outpath, "a+") as fout:
                for line in tqdm(fin, desc="Worker {}".format(workerNum)):
                    fout.write(line)


if __name__ == '__main__':
    processPool = []

    for worker in range(1, 3):
        p = Process(target=run, args=(worker, worker))
        p.start()
        processPool.append(p)
    for p in processPool:
        p.join()
