import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, LineByLineTextDataset, \
    DataCollatorForLanguageModeling, T5Tokenizer, T5Config, T5ForConditionalGeneration
from scipy.special import softmax
import numpy as np


tokenizer = T5Tokenizer.from_pretrained('t5-base')
config = T5Config.from_pretrained('t5-base')
model = T5ForConditionalGeneration.from_pretrained(
            "../model/t5-base/model.ckpt-1004000", from_tf=True, config=config)
# tokenizer = T5Tokenizer.from_pretrained('t5-base')
# model = GPT2LMHeadModel.from_pretrained("results")
doc_text = ["D: Hello I'm a single sentence. Q:", "D: Hello I'm: Q:"]
querys = ["yes i am", "yes i am"]
encoded_encoder_inputs = tokenizer(doc_text, padding=True, truncation=True, return_tensors="pt")
encoded_decoder_inputs = tokenizer(querys, padding=True, truncation=True, return_tensors="pt")
input_ids = tokenizer(doc_text, padding=True, truncation=True, return_tensors="pt")
decoder_input_ids = encoded_decoder_inputs["input_ids"][0]
print(decoder_input_ids)
with torch.no_grad():
    outputs = model.forward(input_ids=encoded_encoder_inputs["input_ids"],
                            labels=encoded_decoder_inputs["input_ids"],
                            attention_mask=encoded_encoder_inputs["attention_mask"])
    batch_logits = outputs[1]
    for logits in batch_logits:
        distributions = softmax(logits.numpy(), axis=1)
        for i in distributions:
            print(np.sum(i))
# outputs = model.generate(
#     input_ids=input_ids,
#     max_length=64,
#     do_sample=True,
#     top_k=10,
#     num_return_sequences=3)
#
# for i in range(3):
#     print(f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')