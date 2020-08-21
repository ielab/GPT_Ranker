from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2-large')

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
print(tokenizer.decode(encoded_input["input_ids"]))