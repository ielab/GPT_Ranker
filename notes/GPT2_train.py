from transformers import AutoTokenizer, Trainer, TrainingArguments, GPT2LMHeadModel, LineByLineTextDataset, \
    DataCollatorForLanguageModeling

import math

tokenizer = AutoTokenizer.from_pretrained("results")

# model = GPT2LMHeadModel.from_pretrained('gpt2-large')
model = GPT2LMHeadModel.from_pretrained("results")
special_token_dict = {
    'pad_token': '<PAD>',
}

num_added_toks = tokenizer.add_special_tokens(special_token_dict)
model.resize_token_embeddings(len(tokenizer))

train_set = LineByLineTextDataset(tokenizer=tokenizer, file_path="train.txt", block_size=None)

data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False)
#
training_args = TrainingArguments(
    output_dir='./results1',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=2,  # batch size per device during training
    per_device_eval_batch_size=2,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    do_eval=True,
    logging_dir='./logs',            # directory for storing logs
)

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    data_collator=data_collator,
    train_dataset=train_set,         # training dataset
    eval_dataset=train_set,        # evaluation dataset
    prediction_loss_only=True,
)

while True:
    trainer.train()
    eval_output = trainer.evaluate()
    perplexity = math.exp(eval_output["eval_loss"])
    print("perplexity:", perplexity)
    if perplexity < 1.1:
        trainer.save_model()
        tokenizer.save_pretrained(training_args.output_dir)
        break
