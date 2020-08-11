from transformers import *

"""
THIS FILE MAINLY HANDLES THE GPT-2 RELATED WORK
"""


class GPT2:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2")

    def prediction(self, document, queryWord):
        pass
