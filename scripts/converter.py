from tqdm import tqdm

# Convert GPT_Ranker res file into trec res file format
with open("result/pass_rerank/t5/t5_rerank_1000-1_lower.res", "r") as f:
    lines = f.readlines()
    for line in tqdm(lines):
        segs = line.rstrip().split("\t")
        with open("result/pass_rerank/t5/t5_rerank_1000_lower_trec.res", "a+") as ff:
            ff.write("{} Q0 {} {} {} {}\n".format(
                segs[0], segs[1], segs[2], segs[3], "t5_1000_lower"))
        ff.close()
f.close()

#########################

# Convert the anserini res file into trec res file format
# with open("data/pass_rerank/run.msmarco-passage.dev.small.tsv", "r") as f:
#     lines = f.readlines()
#     dic = {}
#     for line in tqdm(lines):
#         segs = line.rstrip().split("\t")
#         if segs[0] not in dic:
#             dic[segs[0]] = 0
#         else:
#             continue
#
#     for l in tqdm(lines):
#         s = l.rstrip().split("\t")
#         dic[s[0]] = dic[s[0]] + 1
#
#     for ll in tqdm(lines):
#         ss = ll.rstrip().split("\t")
#         newline = "{} Q0 {} {} {} {}\n".format(ss[0], ss[1], ss[2], dic[ss[0]], "bm25_tuned")
#         dic[ss[0]] = dic[ss[0]] - 1
#         with open("data/pass_rerank/bm25_tuned_trec.res", "a+") as ff:
#             ff.write(newline)
