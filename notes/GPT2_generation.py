
from transformers import AutoTokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, LineByLineTextDataset, \
    DataCollatorForLanguageModeling


tokenizer = AutoTokenizer.from_pretrained("results")
model = GPT2LMHeadModel.from_pretrained("results")
doc_text = "D: Hello I'm a single sentence. Q:"

input_ids = tokenizer.encode(doc_text, return_tensors='pt')
outputs = model.generate(
    input_ids=input_ids,
    max_length=64,
    do_sample=True,
    top_k=10,
    num_return_sequences=3)

for i in range(3):
    print(f'sample {i + 1}: {tokenizer.decode(outputs[i], skip_special_tokens=True)}')