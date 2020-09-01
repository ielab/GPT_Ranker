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
- [Google Pegasus](https://github.com/google-research/pegasus)
    - [Pegasus Paper](https://arxiv.org/pdf/1912.08777.pdf)
- [Facebook BART](https://github.com/pytorch/fairseq/tree/master/examples/bart)
    - [BART Paper](https://arxiv.org/pdf/1910.13461.pdf)
- [RoBERTa](https://pytorch.org/hub/pytorch_fairseq_roberta/)
    - [RoBERTa Paper](https://arxiv.org/pdf/1907.11692.pdf)

### Traditional Models

- [Query Likelihood Language Model](https://en.wikipedia.org/wiki/Query_likelihood_model)

### Tools

- [Google Cloud TPU](https://cloud.google.com/tpu/docs/quickstart)

- - -
## To Clone this repo

```
git clone --recursive-submodules git@github.com:ielab/GPT_Ranker.git
```

- - -
## The Recall/MRR@10 evaluation for msmarco doc dev (5193 queries)

|  Model Name           |  Top K                  |Recall   | MRR@100   |
|:---------------------:|:-----------------------:|:-------:|:---------:|
| T5-Base               |Top 100<br/>(Tuned BM25) |         |           |
|                       |Top 200<br/>(Tuned BM25) |         |           |
|                       |Top 500<br/>(Tuned BM25) |         |           |
|                       |Top 1000<br/>(Tuned BM25)|         |           |
| GPT-2                 |Top 100<br/>(Tuned BM25) |         |           |
|                       |Top 200<br/>(Tuned BM25) |         |           |
|                       |Top 500<br/>(Tuned BM25) |         |           |
|                       |Top 1000<br/>(Tuned BM25)|         |           |
|BM25 Initial Retrieval<br/>(Tuned `k1=3.44 b=0.87`) |  N/A   | `R@5 0.4024`<br/>`R@10 0.4946`<br/>`R@15 0.5640`<br/>`R@20 0.6095`<br/>`R@30 0.6649`<br/>`R@100 0.7874`<br/>`R@200 0.8373`<br/>`R@500 0.8850`<br/>`R@1000 0.9187` | `0.27880910` |
|MS MARCO Top 1000<br/>(Provided Run)|  N/A   |         |           |

- - -
## The Recall/MRR@10 evaluation for msmarco passage dev small (6980 queries)

|   Model Name          |Top K         |Recall   | MRR@10   |
| :--:                  | :---:        |:-----:  | :-----:  |
| T5-Base               |Top 100<br/>(Tuned BM25)| `R@5 0.4294`<br/>`R@10 0.5321`<br/>`R@15 0.5778`<br/>`R@20 0.6076`<br/>`R@30 0.6354`<br/>`R@100 0.6701`<br/>`R@200 0.6701`<br/>`R@500 0.6701`<br/>`R@1000 0.6701` | `0.28134977` |
|                       | Top 200<br/>(Tuned BM25)| `R@5 0.4413`<br/>`R@10 0.5496`<br/>`R@15 0.6037`<br/>`R@20 0.6371`<br/>`R@30 0.6719`<br/>`R@100 0.7333`<br/>`R@200 0.7383`<br/>`R@500 0.7383`<br/>`R@1000 0.7383` | `0.286065970`|
|                       | Top 500<br/>(Tuned BM25)| `R@5 0.4502`<br/>`R@10 0.5630`<br/>`R@15 0.6195`<br/>`R@20 0.6561`<br/>`R@30 0.6987`<br/>`R@100 0.7812`<br/>`R@200 0.8040`<br/>`R@500 0.8116`<br/>`R@1000 0.8116` | `0.2904781`  |
|                       |Top 1000<br/>(Tuned BM25)| `R@5 0.4553`<br/>`R@10 0.5708`<br/>`R@15 0.6318`<br/>`R@20 0.6700`<br/>`R@30 0.7148`<br/>`R@100 0.8093`<br/>`R@200 0.8390`<br/>`R@500 0.8553`<br/>`R@1000 0.8573` | `0.2920570`  |
|                       |Top 1000<br/>(Tuned BM25 +<br/>Lowercase P)| `R@5 0.4546`<br/>`R@10 0.5733`<br/>`R@15 0.6346`<br/>`R@20 0.6679`<br/>`R@30 0.7133`<br/>`R@100 0.8095`<br/>`R@200 0.8384`<br/>`R@500 0.8552`<br/>`R@1000 0.8573` | `0.2945217`  |
|                       |Top 1000</br>(docTTTTTquery init run)| `R@5 0.4665`<br>`R@10 0.5857`<br/>`R@15 0.6537`<br/>`R@20 0.6968`<br/>`R@30 0.7485`<br/>`R@100 0.8643`<br/>`R@200 0.9089`<br/>`R@500 0.9404`<br/>`R@1000 0.9471` | `0.29763735` |
| GPT-2                 |Top 100<br/>(Tuned BM25)|         |          |
|                       |Top 200<br/>(Tuned BM25)|         |          |
|                       |Top 500<br/>(Tuned BM25)|         |          |
|                       |Top 1000<br/>(Tuned BM25)|         |          |
|BM25 Initial Retrieval<br/>(Tuned `k1=0.82 b=0.68`) |    N/A       | `R@5 0.2944`<br/>`R@10 0.3916`<br/>`R@15 0.4459`<br/>`R@20 0.4842`<br/>`R@30 0.5307`<br/>`R@100 0.6701`<br/>`R@200 0.7383`<br/>`R@500 0.8116`<br/>`R@1000 0.8573` | `0.187412`   |
|MS MARCO Top 1000<br/>(Provided Run)|    N/A       | `R@5 0.0093`<br/>`R@10 0.0150`<br/>`R@15 0.0196`<br/>`R@20 0.0224`<br/>`R@30 0.0270`<br/>`R@100 0.1026`<br/>`R@200 0.1641`<br/>`R@500 0.3893`<br/>`R@1000 0.8140` |  `0.00456946`|
|docTTTTTquery Top 1000<br/>(40 Samples)|    N/A       | `R@5 0.4244`<br/>`R@10 0.5411`<br/>`R@15 0.6033`<br/>`R@20 0.6484`<br/>`R@30 0.6987`<br/>`R@100 0.8190`<br/>`R@200 0.8688`<br/>`R@500 0.9164`<br/>`R@1000 0.9471` | `0.2767497` |

## For example in passage task:

For query:
```
how many tables can sql server join
```

Our model ranks this at top 1:
```
id: 7485889
contents: How many tables can I have in 1 Sql Azure Database. I know in Sql Server, Tables per database Limited by number of objects in a database, Database objects include objects such as tables, views, stored procedures, user-defined functions, triggers, rules, defaults, and constraints. The sum of the number of objects in a database cannot exceed 2,147,483,647..
```

The actual relevant document is (we rank this at 949, BM25 rank this at 300):
```
id: 7485894
contents: SQL JOIN. A JOIN clause is used to combine rows from two or more tables, based on a related column between them. Let's look at a selection from the Orders table: Then, look at a selection from the Customers table: Notice that the CustomerID column in the Orders table refers to the CustomerID in the Customers table. The relationship between the two tables above is the CustomerID column. Then, we can create the following SQL statement (that contains an INNER JOIN), that selects records that have matching values in both tables: Example SELECT Orders.OrderID, Customers.CustomerName, Orders.OrderDate
```
- - -

For query:
```
what does it mean when you dream about babies
```

Our model ranks this at top 1:
```
id: 6680536
contents: A baby in general. Dreams that include babies are positive signs. Dreaming about interacting with a baby or simply seeing a baby in a dream can mean that pleasant surprises and fortuitous occurrences are about to occur in your life. This dream doesn't specify what, but something unexpectedly good is on your horizon.
```

The actual relevant document is (we rank this at 2, BM25 rank this at 993):
```
id: 7551052
contents: To see a baby in your dream signifies innocence, warmth and new beginnings. Babies symbolize something in your own inner nature that is pure, vulnerable, helpless and/or uncorrupted. If you dream that the baby is smiling at you, then it suggests that you are experiencing pure joy.
```

BM25 ranks this at top 1 (we rank this at 47):
```
id: 3347686
contents: Health related question in topics Psychology .We found some answers as below for this question What does it mean when you dream about somebody having a baby,you can compare them. When you dream of someone having a baby, it means new beginnings. It also means hidden potential related to you is being released. [ Source: http://www.chacha.com/question/what-does-it-mean-when-you-dream-about-somebody-having-a-baby ] More Answers to What does it mean when you dream about somebody having a baby.
```
- - -

Comparison with BM25 model:

![Rank Comparison](./images/t5_1000_docTquery_vs_bm25_tuned.png)

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
