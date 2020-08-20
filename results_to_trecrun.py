import os


def results_to_trecrun(folder_path, output):
    files = os.listdir(folder_path)
    trecrun_file = open(output, "a")

    for f in files:
        if f == '.DS_Store':
            pass
        else:
            file_path = folder_path+'/'+f
            with open(file_path) as f:
                for line in f:
                    line = line.strip()
                    topicid, docid, rank, score = line.split("\t")
                    line = "{} Q0 {} {} {} ielab_bm25T5QLM".format(topicid, docid, rank, score)
                    trecrun_file.write(line+"\n")

results_to_trecrun("/Users/s4416495/Downloads/trec_cast_2020/result/t5", "/Users/s4416495/Downloads/trec_cast_2020/result/ielab_bm25T5QLM.res")