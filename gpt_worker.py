from transformers import *


class GPT2:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelWithLMHead.from_pretrained("gpt2")

    def prediction(self, document, queryWord):
        pass
