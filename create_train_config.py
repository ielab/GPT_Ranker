import json
args_dict = {
  "num_cores": 8,
  "model_type": "gpt2",
  "model_name_or_path": 'gpt2-large',
  "train_data_file": "gs://ielab-msmarco/dataset/pass.train.tsv",
  "output_dir": 'gs://ielab-msmarco/models/gpt2_pass/',
  "overwrite_output_dir": True,
  "per_device_train_batch_size": 8,
  "per_device_eval_batch_size": 8,
  "learning_rate": 5e-5,
  "tpu_num_cores": 8,
  "num_train_epochs": 4,
  "do_train": True
}

with open('args.json', 'w') as f:
  json.dump(args_dict, f)