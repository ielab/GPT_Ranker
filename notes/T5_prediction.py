import torch
from transformers import *
import numpy as np
from scipy.special import softmax
torch.manual_seed(0)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
    "../model/t5-base/model.ckpt-1004000", from_tf=True, config=config)
model.eval()

def prediction(documents, query):
    documents = [document + ' </s>' for document in documents]
    querys = [query] * len(documents)
    encoded_encoder_inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    encoded_decoder_inputs = tokenizer(querys, padding=True, truncation=True, return_tensors="pt")
    decoder_input_ids = encoded_decoder_inputs["input_ids"][0]
    scores = []
    with torch.no_grad():
        outputs = model(input_ids=encoded_encoder_inputs["input_ids"],
                             labels=encoded_decoder_inputs["input_ids"],
                             attention_mask=encoded_encoder_inputs["attention_mask"])
        batch_logits = outputs[1]
        batch_logits = batch_logits.cpu()
        for logits in batch_logits:
            distributions = softmax(logits.numpy(), axis=1)
            prob = []
            for index, val in enumerate(decoder_input_ids):
                prob.append(distributions[index][val])
            score = np.sum(np.log(prob))
            scores.append(score)
    return scores

documents = ["A baby in general. Dreams that include babies are positive signs. Dreaming about interacting with a baby or simply seeing a baby in a dream can mean that pleasant surprises and fortuitous occurrences are about to occur in your life. This dream doesn't specify what, but something unexpectedly good is on your horizon.",
             "dreams To see a baby in your dream signifies innocence, warmth and new beginnings. Babies symbolize something in your own mean inner nature that is pure, vulnerable, abhout helpless and/or uncorrupted. If you dream that the baby is smiling at you, then it suggests that you are experiencing pure joy. babies"]
query = "what does it mean when you dream about babies"
print(prediction(documents, query))
