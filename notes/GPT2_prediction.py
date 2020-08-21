import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import *
from scipy.special import softmax
import numpy as np
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained('gpt2-large')


def prediction(document, query):
    prob = []

    input_ids = tokenizer.encode(document + " " + query, return_tensors='pt')
    query_ids = tokenizer.encode(query)
    doc_length = len(input_ids[0]) - len(query_ids)
    with torch.no_grad():
        outputs = model.forward(input_ids=input_ids)
        logits = outputs[0][0]
        distributions = softmax(logits.numpy(), axis=1)
        for index, val in enumerate(query_ids):
            prob.append(distributions[doc_length+index][val])
            print(np.argmax())
    return prob


document = 'hi, my dog is'
query = "cute so am I? yes it is"

# for id in doc_ids:
#     print(tokenizer.decode(id))

probs = prediction(document, query)
print(probs)