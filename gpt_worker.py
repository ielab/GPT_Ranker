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


class GPT2:

    def __init__(self, model_dir=None):
        if model_dir is not None:
            self.tokenizer = T5Tokenizer.from_pretrained(model_dir)
            config = T5Config.from_pretrained(model_dir)
            self.model = T5ForConditionalGeneration.from_pretrained(
                'model.ckpt-1004000', from_tf=True, config=config)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
            self.model = AutoModelWithLMHead.from_pretrained("gpt2-large")

        self.model.to(DEVICE)

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
