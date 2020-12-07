import os

command = 'python run_classifier.py --data_dir yahoo ' \
          '--bert_model bert-base-uncased ' \
          '--task_name ag --output_dir results/yahoo --cache_dir pytorch_cache --do_train  --do_eval --do_lower_case '

os.system(command)
