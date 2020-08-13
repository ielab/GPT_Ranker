import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import *
from scipy.special import softmax

"""
THIS FILE MAINLY HANDLES THE GPT-2 RELATED WORK
"""


class GPT2:

    def __init__(self):
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
