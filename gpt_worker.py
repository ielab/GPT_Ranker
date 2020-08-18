import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import *
from scipy.special import softmax
import numpy
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

"""
THIS FILE MAINLY HANDLES THE GPT-2 RELATED WORK
"""


class T5:

    def __init__(self, model_dir):
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
            predictions = self.model(input_ids=encoder_input_ids, labels=decoder_input_ids)
            results = predictions[1][0]
            result = softmax(results.numpy(), axis=1)
            for index, val in enumerate(decoder_input_ids[0]):
                prob.append(result[index][val])
        return prob

    def getPredictionScore(self, document, query):
        # Get the prediction score from the GPT-2 model
        prob = self.prediction(document, query)
        score = numpy.sum(numpy.log(prob))
        return score


class GPT2:

    def __init__(self,):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2-large")

    def prediction(self, document, queryWords, query):
        prob = []

        documentLength = len(document.split(" "))

        indexed_tokens = self.tokenizer.encode(document+" "+query)
        token_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            predictions = self.model(token_tensor)
            results = predictions[0]
            temp = results[0, documentLength - 1:-1, :]
            result = softmax(temp.numpy(), axis=1)
            words = self.tokenizer.encode(queryWords)
            for index, val in enumerate(words):
                prob.append(result[index][val])

        return prob

    def getPredictionScore(self, document, query):
        # Get the prediction score from the GPT-2 model
        queryTokens = query.split(" ")

        prob = self.prediction(document, queryTokens, query)
        score = numpy.sum(numpy.log(prob))
        return score
