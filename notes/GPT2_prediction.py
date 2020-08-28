import torch
from transformers import *
torch.manual_seed(0)
tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
    "../model/t5-base/model.ckpt-1004000", from_tf=True, config=config)
model.eval()

def prediction(documents, query):
    querys = [query] * len(documents)
    encoded_encoder_inputs = tokenizer(documents, padding=True, truncation=True, return_tensors="pt")
    encoded_decoder_inputs = tokenizer(querys, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(input_ids=encoded_encoder_inputs["input_ids"],
                        labels=encoded_decoder_inputs["input_ids"],
                        attention_mask=encoded_encoder_inputs["attention_mask"])
        batch_logits = outputs[1]
        print(batch_logits)


documents = ['MY DOG is Cute!'.lower(), "so am I."]
query = "who am I?"
prediction(documents, query)
