import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import T5Tokenizer, T5Config, T5ForConditionalGeneration, GPT2Tokenizer, GPT2LMHeadModel
from scipy.special import softmax
import numpy

torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

"""
THIS FILE MAINLY HANDLES THE RANKER RELATED WORK
"""


class T5:
    def __init__(self, model_dir):
        self.name = "T5"

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        config = T5Config.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_dir, from_tf=True, config=config)
        self.model.eval()
        self.model.to(DEVICE)
        self.softmax = torch.nn.Softmax(dim=1).to(DEVICE)

    def predict(self, document, query, conf):
        if "CHUNK_SIZE" in conf:
            scores = []
            segments = document.split(" ")
            x = 0
            y = len(segments)
            for i in range(x, y, conf["CHUNK_SIZE"]):
                x = i
                temp = ' '.join(segments[x:x + conf["CHUNK_SIZE"]])
                temp += ' </s>'
                prob = []
                encoder_input_ids = self.tokenizer.encode(document, return_tensors='pt').to(DEVICE)
                decoder_input_ids = self.tokenizer.encode(query, return_tensors='pt').to(DEVICE)

                with torch.no_grad():
                    outputs = self.model(input_ids=encoder_input_ids, labels=decoder_input_ids).to(DEVICE)
                    logits = outputs[1][0]
                    distributions = softmax(logits.numpy(), axis=1)
                    for index, val in enumerate(decoder_input_ids[0]):
                        prob.append(distributions[index][val])
                score = numpy.sum(numpy.log10(prob))
                scores.append(score)
            return max(scores)
        else:
            document += ' </s>'
            prob = []
            encoder_input_ids = self.tokenizer.encode(document, return_tensors='pt').to(DEVICE)
            decoder_input_ids = self.tokenizer.encode(query, return_tensors='pt').to(DEVICE)

            with torch.no_grad():
                outputs = self.model(input_ids=encoder_input_ids, labels=decoder_input_ids)
                logits = outputs[1][0]
                distributions = softmax(logits.numpy(), axis=1)
                for index, val in enumerate(decoder_input_ids[0]):
                    prob.append(distributions[index][val])
            score = numpy.sum(numpy.log10(prob))
            return score

    def tokenize(self, text):
        encoded_inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        return encoded_inputs

    def batchPredict(self, documents, encoded_decoder_inputs, conf):
        documents = [document + ' </s>' for document in documents]
        # querys = [query] * len(documents)
        encoded_encoder_inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors="pt").to(
            DEVICE)
        # encoded_decoder_inputs = self.tokenizer(querys, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        decoder_input_ids = encoded_decoder_inputs["input_ids"][0]
        scores = []
        with torch.no_grad():
            outputs = self.model(input_ids=encoded_encoder_inputs["input_ids"],
                                 labels=encoded_decoder_inputs["input_ids"],
                                 attention_mask=encoded_encoder_inputs["attention_mask"])
            batch_logits = outputs[1]
            # batch_logits = batch_logits.cpu()

            for logits in batch_logits:
                # distributions = softmax(logits.numpy(), axis=1)
                distributions = self.softmax(logits)
                prob = []
                for index, val in enumerate(decoder_input_ids):
                    prob.append(distributions[index][val])
                score = numpy.sum(numpy.log10(prob))
                scores.append(score)
        return scores


class GPT2:

    def __init__(self, model_dir=None):
        self.name = "GPT2"
        if model_dir is not None:
            pass
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-large')
            self.model.to(DEVICE)

    def predict(self, document, query, conf):
        if "CHUNK_SIZE" in conf:
            scores = []
            segments = document.split(" ")
            x = 0
            y = len(segments)
            for i in range(x, y, conf["CHUNK_SIZE"]):
                x = i
                temp = ' '.join(segments[x:x + conf["CHUNK_SIZE"]])
                prob = []
                input_ids = self.tokenizer.encode(temp + " " + query, return_tensors='pt').to(DEVICE)
                query_ids = self.tokenizer.encode(query).to(DEVICE)
                doc_length = len(input_ids[0]) - len(query_ids)
                with torch.no_grad():
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs[0][0]
                    distributions = softmax(logits.numpy(), axis=1)
                    for index, val in enumerate(query_ids):
                        prob.append(distributions[doc_length + index][val])
                score = numpy.sum(numpy.log10(prob))
                scores.append(score)
            return max(scores)
        else:
            prob = []
            input_ids = self.tokenizer.encode(document + " " + query, return_tensors='pt').to(DEVICE)
            query_ids = self.tokenizer.encode(query).to(DEVICE)
            doc_length = len(input_ids[0]) - len(query_ids)
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs[0][0]
                distributions = softmax(logits.numpy(), axis=1)
                for index, val in enumerate(query_ids):
                    prob.append(distributions[doc_length + index][val])
            score = numpy.sum(numpy.log10(prob))
            return score
