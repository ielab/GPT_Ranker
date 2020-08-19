import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import *
from scipy.special import softmax
import numpy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
THIS FILE MAINLY HANDLES THE RANKER RELATED WORK
"""


class Ranker:
    def getPredictionScore(self, document, query):
        prob = self.prediction(document, query)
        score = numpy.sum(numpy.log(prob))
        return score

    def prediction(self, document, query):
        pass


class T5(Ranker):

    def __init__(self, model_dir):
        self.name = "T5"

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')
        config = T5Config.from_pretrained('t5-base')
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_dir, from_tf=True, config=config)
        self.model.to(DEVICE)

    def prediction(self, document, query):
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
        return prob


class GPT2(Ranker):

    def __init__(self, model_dir=None):
        self.name = "GPT2"
        if model_dir is not None:
            pass
        else:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
            self.model = GPT2LMHeadModel.from_pretrained('gpt2-large')
            self.model.to(DEVICE)

    def prediction(self, document, query):
        prob = []

        input_ids = self.tokenizer.encode(document + " " + query, return_tensors='pt')
        query_ids = self.tokenizer.encode(query)
        doc_length = len(input_ids[0]) - len(query_ids)
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
            logits = outputs[0][0]
            distributions = softmax(logits.numpy(), axis=1)
            for index, val in enumerate(query_ids):
                prob.append(distributions[doc_length + index][val])
        return prob
