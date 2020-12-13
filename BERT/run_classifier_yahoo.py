import os

command = 'python run_classifier.py --data_dir yahoo ' \
          '--bert_model bert-base-uncased --max_seq_length 256 --train_batch_size 16 ' \
          '--task_name fake --output_dir results/yahoo --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '

os.system(command)
