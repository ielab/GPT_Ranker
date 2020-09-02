from tqdm import tqdm
import json
from sentence_splitter import SentenceSplitter

collection = [
      "../data/pass_rerank/collection_jsonl/docs00.json",
      "../data/pass_rerank/collection_jsonl/docs01.json",
      "../data/pass_rerank/collection_jsonl/docs02.json",
      "../data/pass_rerank/collection_jsonl/docs03.json",
      "../data/pass_rerank/collection_jsonl/docs04.json",
      "../data/pass_rerank/collection_jsonl/docs05.json",
      "../data/pass_rerank/collection_jsonl/docs06.json",
      "../data/pass_rerank/collection_jsonl/docs07.json",
      "../data/pass_rerank/collection_jsonl/docs08.json"
    ]

splitter = SentenceSplitter(language='en')

totalDocument = 0
totalSentence = 0
longestDocument = 0
shortestDocument = 999999

for path in tqdm(collection, desc="Processing Collection"):
    with open(path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines, desc="Processing Lines"):
            totalDocument += 1
            content = json.loads(line)["contents"]
            sentences = splitter.split(content)
            totalSentence += len(sentences)
            if len(sentences) > longestDocument:
                longestDocument = len(sentences)
            if len(sentences) < shortestDocument:
                shortestDocument = len(sentences)

print("Total Documents: " + str(totalDocument))
print("Total Sentences: " + str(totalSentence))
print("Longest Document Has " + str(longestDocument) + " Sentences")
print("Shortest Document Has " + str(shortestDocument) + " Sentences")
print("Average " + str(totalSentence/totalDocument) + " Sentences Per Document")

# Total Documents: 3213835
# Total Sentences: 168483433
# Longest Document Has 26680 Sentences
# Shortest Document Has 0 Sentences
# Average 52.424419112991174 Sentences Per Document
