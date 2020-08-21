
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from torch.utils.data import TensorDataset, random_split
import torch



model = GPT2LMHeadModel.from_pretrained('gpt2-large')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large')

special_token_dict = {
    'bos_token': '<BOS>',
    'eos_token': '<EOS>',
    'pad_token': '<PAD>'
}
num_added_toks = tokenizer.add_special_tokens(special_token_dict)
model.resize_token_embeddings(len(tokenizer))

input_ids = []
attention_masks = []
labels = [0, 1]

training_args = TrainingArguments(
    output_dir='./model/gpt-2',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs/gpt2'
)

for sent in ["hello, this is a cat", "hello this is a dog"]:
    encoded_dict = tokenizer.encode_plus(
        sent,
        add_special_tokens=True,
        max_length=64,
        pad_to_max_length=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids.append(encoded_dict['input_ids'])
    attention_masks.append(encoded_dict['attention_mask'])
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(labels)

train_dataset = TensorDataset(input_ids, attention_masks, labels)
eval_dataset = TensorDataset(input_ids, attention_masks, labels)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

trainer.train()
trainer.evaluate()

# https://huggingface.co/transformers/training.html#trainer
# https://towardsdatascience.com/fine-tuning-gpt2-for-text-generation-using-pytorch-2ee61a4f1ba7
# https://github.com/itsuncheng/fine-tuning-GPT2
# https://github.com/itsuncheng/fine-tuning-GPT2/blob/master/preprocessing.ipynb
# https://mccormickml.com/2019/07/22/BERT-fine-tuning/
# https://colab.research.google.com/drive/1pTuQhug6Dhl9XalKB0zUGf4FIdYFlpcX
