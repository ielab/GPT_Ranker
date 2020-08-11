from transformers import *
import torch
from scipy.special import softmax

"""
THIS FILE MAINLY HANDLES THE GPT-2 RELATED WORK
"""


class GPT2:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2-large")

    def prediction(self, document, queryWords):
        prob = []
        # truth = "D: In-home tutors can earn anywhere from $10 to $80 an hour, depending on the type of lesson, the student\u00e2\u0080\u0099s skill and age level and the tutor\u00e2\u0080\u0099s experience. Tutors often charge more for older students or those who require more advanced lessons.\nQ: how much does an average person make for tutoring\n"
        # document = truth + "D: " + document + "\nQ:"
        for each in queryWords:
            indexed_tokens = self.tokenizer.encode(document)
            token_tensor = torch.tensor([indexed_tokens])

            with torch.no_grad():
                predictions = self.model(token_tensor)
                results = predictions[0]
                temp = results[0, -1, :]
                temp = temp.numpy()
                result = softmax(temp)
                word = self.tokenizer.encode(each)[0]
                prob.append(result[word])
                document = document + " " + each

        return prob
