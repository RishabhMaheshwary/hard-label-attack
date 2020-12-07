### Generating Natural Language Attack in Hard Label Black Box Setting

Computing Infrastructure Used: Nvidia GeForce GTX 1080 Ti GPUs, providing 14336 CUDA cores, and 44 GB of GDDR5X VRAM


Requirements
-  Pytorch >= 0.4
-  Tensorflow >= 1.0
-  Numpy
-  Python >= 3.6
- Tensorflow 2.1.0
- TensorflowHub

Dependencies

- Download pretrained target models for each dataset from [here](https://drive.google.com/file/d/1UChkyjrSJAVBpb3DcPwDhZUE4FuL0J25/view?usp=sharing), unzip it.

- Download top 50 synonym file of counter-fitted-vectors from [here](https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view), unzip it and place the txt file in the main directory.

- Download the glove 200 dimensional vectors from [here](https://nlp.stanford.edu/projects/glove/) unzip it.
 
How to Run:
```
-   Use the following command to get the results. 

-For BERT model

```
python3 classification_attack.py \
        --dataset_path path_to_data_samples_to_attack  \
        --target_model Type_of_taget_model (bert,wordCNN,wordLSTM) \
        --counter_fitting_cos_sim_path path_to_top_50_synonym_file \
        --target_dataset dataset_to_attack (imdv,ag,yelp,yahoo,mr) \
        --target_model_path path_to_pretrained_target_model \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses classes_in_the_dataset_to_attack
```
- Example of attacking BERT on IMDB dataset.

```
python3 classification_attack.py \
        --dataset_path data/imdb  \
        --target_model bert \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset imdb \
        --target_model_path BERT/results/imdb \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses 2
```

-   Example of attacking BERT on SNLI dataset. 

```
python3 nli_attack.py \
        --dataset_path data/snli  \
        --target_model bert \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset snli \
        --target_model_path BERT/results/snli \
        --USE_cache_path "nli_cache" \
        --sim_score_window 40
```
The results will be available in **results_hard_label** for classification task and in **results_nli_hard_label** for entailment tasks.
For attacking other target models look at the commands in the ```commands``` folder.
