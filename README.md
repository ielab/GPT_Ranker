# GPT_Ranker

This Branch Is For MSMARCO Passage Reranking Task Using GPT-2 Model


## TREC CAsT Datasets

- MS MARCO: https://microsoft.github.io/msmarco/  
  - Git Repo: https://github.com/microsoft/MSMARCO-Passage-Ranking
- TREC CAR: http://trec-car.cs.unh.edu/datareleases/v2.0-release.html
- castorini docTTTTTquery: https://github.com/castorini/docTTTTTquery#ms-marco-document-dataset

## To Clone this repo

git clone --recursive-submodules git@github.com:ielab/GPT_Ranker.git

## File Structure

_The file structure should be same as this_
```
GPT_Ranker/
+--- anserini/
+--- data/
|    +--- doc_rerank/
|    |    +--- collection_jsonl/
|    |    |    +--- docs00.json
|    |    |    +--- docs01.json
|    |    |    +--- docs02.json
|    |    |    +--- docs03.json
|    |    |    +--- docs04.json
|    |    |    +--- docs05.json
|    |    |    +--- docs06.json
|    |    +--- index/
|    |    |    +--- lucene-index.msmarco-doc.pos+docvectors+rawdocs/
|    |    |    |    +--- HERE CONTAINS THE ANSERINI MSMARCO DOCUMENT INDEX
|    |    +--- qrels/
|    |    |    +--- qrels.msmarco-doc.dev.txt
|    |    +--- query/
|    |    |    +--- doc-msmarco-dev-queries.json
|    |    |    +--- doc-msmarco-test2020-queries.json
|    |    +--- formatted_run.msmarco-doc.dev.bm25.tuned.txt
|    |    +--- formatted_run.msmarco-doc.test.bm25.tuned.txt
|    +--- pass_rerank
|    |    +--- collection_jsonl/
|    |    |    +--- docs00.json
|    |    |    +--- docs01.json
|    |    |    +--- docs02.json
|    |    |    +--- docs03.json
|    |    |    +--- docs04.json
|    |    |    +--- docs05.json
|    |    |    +--- docs06.json
|    |    |    +--- docs07.json
|    |    |    +--- docs08.json
|    |    +--- index/
|    |    |    +--- lucene-index-msmarco/
|    |    |    |    +--- HERE CONTAINS THE ANSERINI MSMARCO PASSAGE INDEX
|    |    +--- qrels/
|    |    |    +--- qrels.dev.small.tsv
|    |    |    +--- pass-qrels.msmarco-dev.full.txt
|    |    +--- query/
|    |    |    +--- pass-queries.dev.json
|    |    |    +--- pass-queries.eval.json
|    |    |    +--- pass-query-dev.small.json
|    |    +--- run.msmarco-passage.dev.small.tsv
|    |    +--- run.msmarco-passage.dev.full.tsv
|    |    +--- run.msmarco-passage.dev.eval.tsv
|    +--- pass_train/
|    |    +--- doc_query_pairs.train.tsv
|    +--- doc_train/
|    |    +--- doc_query_pairs.train.tsv
+--- logs/
|    +--- gpt2/
|    |    +--- CONTAIN GPT2 TRAINING LOGS
|    +--- t5/
|    |    +--- CONTAIN T5 TRAINING LOGS
+--- model/
|    +--- gpt-2/
|    |    +--- CONTAIN PRETRAINED GPT-2 (UNTUNED)
|    +--- t5-base/
|    |    +--- CONTAIN T5 FINE TUNED ON MSMARCO PASSAGE
|    +--- t5-base-tuned/
|    |    +--- tuned_on_doc/
|    |    |    +--- CONTAIN T5 FINE TUNED ON MSMARCO DOCUMENT
|    +--- gpt-2-tuned/
|    |    +--- tuned_on_pass/
|    |    |    +--- CONTAIN GPT-2 FINE TUNED ON MSMARCO PASSAGE
|    |    +--- tuned_on_doc/
|    |    |    +--- CONTAIN GPT-2 FINE TUNED ON MSMARCO DOCUMENT
+--- result/
|    +--- doc_rerank/
|    |    +--- gpt2/
|    |    +--- t5/
|    +--- pass_rerank/
|    |    +--- gpt2/
|    |    +--- t5/
+--- notes/
+--- config.json
+--- fine_tuning.py
+--- helper.py
+--- main.py
+--- middleware.py
+--- passage_msmarco_eval.py
+--- doc_msmarco_eval.py
+--- ranker.py
+--- anserini_retriever.py
+--- README.md
+--- .gitignore
```
