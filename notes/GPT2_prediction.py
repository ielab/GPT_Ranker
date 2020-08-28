import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
from transformers import *
from scipy.special import softmax
import numpy as np

torch.manual_seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# tokenizer = T5ForConditionalGeneration.from_pretrained('t5-base')
# model = T5Tokenizer.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
                "../model/t5-base/model.ckpt-1004000", from_tf=True, config=config)



def prediction(documents, query):
    documents = [document + ' </s>' for document in documents]
    querys = [query] * len(documents)
    encoded_encoder_inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt").to(
        DEVICE)
    encoded_decoder_inputs = tokenizer(querys, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    decoder_input_ids = encoded_decoder_inputs["input_ids"][0]
    scores = []
    with torch.no_grad():
        outputs = model(input_ids=encoded_encoder_inputs["input_ids"],
                             labels=encoded_decoder_inputs["input_ids"],
                             attention_mask=encoded_encoder_inputs["attention_mask"])
        batch_logits = outputs[1]
        print(batch_logits)
        batch_logits = batch_logits.cpu()

        for logits in batch_logits:
            distributions = softmax(logits.numpy(), axis=1)
            prob = []
            for index, val in enumerate(decoder_input_ids):
                prob.append(distributions[index][val])
            score = np.sum(np.log(prob))
            scores.append(score)
    return scores


document = ['hi, my dog is', "yes it is"]
query = "cute so am I? yes it is"

# for id in doc_ids:
#     print(tokenizer.decode(id))

probs = prediction(document, query)
