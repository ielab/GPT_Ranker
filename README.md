# GPT_Ranker

This Repo Is For MSMARCO Document/Passage Reranking Task Using GPT-2/T5 Model
- - -

## Related Links

### Dataset

- [MS MARCO Home Page](https://microsoft.github.io/msmarco/)
- [Passage Ranking Git Repo](https://github.com/microsoft/MSMARCO-Passage-Ranking)
- [Document Ranking Git Repo](https://github.com/microsoft/MSMARCO-Document-Ranking)
- [Jimmy Lin Anserini Passage Retrieval](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-passage.md)
- [Jimmy Lin Anserini Document Retrieval](https://github.com/castorini/anserini/blob/master/docs/experiments-msmarco-doc.md)
- [TREC 2020 Deep Learning Task](https://microsoft.github.io/TREC-2020-Deep-Learning/)

### Models

- [castorini docTTTTTquery](https://github.com/castorini/docTTTTTquery#ms-marco-document-dataset)
- [Google T5](https://github.com/google-research/text-to-text-transfer-transformer)
- [OpenAI GPT-2](https://github.com/openai/gpt-2)
- [Hugging Face Framework](https://github.com/huggingface/transformers)

### Traditional Models

- [Query Likelihood Language Model](https://en.wikipedia.org/wiki/Query_likelihood_model)

### Tools

[Google Cloud TPU](https://cloud.google.com/tpu/docs/quickstart)

- - -
## To Clone this repo

```
git clone --recursive-submodules git@github.com:ielab/GPT_Ranker.git
```

- - -
## The Recall/MRR@10 evaluation for msmarco passage dev small bm25 tuned initial retrieval

__*Recall*:__

```
recall_5              	all	0.2944
recall_10             	all	0.3916
recall_15             	all	0.4459
recall_20             	all	0.4842
recall_30             	all	0.5307
recall_100            	all	0.6701
recall_200            	all	0.7383
recall_500            	all	0.8116
recall_1000           	all	0.8573
```

__*MRR @10:*__

```
#####################
MRR @100: 0.278809
QueriesRanked: 5193
#####################
```
- - -
## The Recall/MRR@10 evaluation for msmarco doc dev bm25 tuned initial retrieval

__*Recall*:__

```
recall_5              	all	0.4140
recall_10             	all	0.5207
recall_15             	all	0.5860
recall_20             	all	0.6307
recall_30             	all	0.6844
recall_100            	all	0.8065
recall_200            	all	0.8552
recall_500            	all	0.9062
recall_1000           	all	0.9326
```

__*MRR @10*:__

```
#####################
MRR @10: 0.187412
QueriesRanked: 6980
#####################
```

- - -
## File Structure

__*The file structure should be same as this*:__

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
+--- SOME OTHER FILES
```
