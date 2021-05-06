### Generating Natural Language Attacks in a Hard Label Black Box Setting

This repository contains source code for the research work described in our AAAI 2021 paper: 

[Generating Natural Language Attacks in a Hard Label Black Box Setting](https://www.researchgate.net/publication/347304785_Generating_Natural_Language_Attacks_in_a_Hard_Label_Black_Box_Setting)

#### The hard label attack has also been implemented in [TextAttack](https://github.com/RishabhMaheshwary/TextAttack/tree/hard_label_attack) library.

Follow these steps to run the attack from the library:

1. Fork the [repository](https://github.com/RishabhMaheshwary/TextAttack/tree/hard_label_attack)

2. Run the following command to install it.

   ```bash
   $ cd TextAttack
   $ pip install -e . ".[dev]"
   
2. Run the following command to attack `bert-base-uncased` trained on `MovieReview` dataset.

   ```bash
   $ textattack attack --recipe hard-label-attack --model bert-base-uncased-mr --num-examples 100

Take a look at the `models` directory in [TextAttack](https://github.com/RishabhMaheshwary/TextAttack/tree/hard_label_attack) to run the attack across any dataset and any target model.

#### Instructions for running the attack from this repository.

#### Requirements
-  Pytorch >= 0.4
-  Tensorflow >= 1.0
-  Numpy
-  Python >= 3.6
- Tensorflow 2.1.0
- TensorflowHub

#### Download Dependencies

- Download pretrained target models for each dataset [bert](https://drive.google.com/file/d/1UChkyjrSJAVBpb3DcPwDhZUE4FuL0J25/view?usp=sharing), [lstm](https://drive.google.com/drive/folders/1nnf3wrYBrSt6F3Ms10wsDTTGFodrRFEW?usp=sharing), [cnn](https://drive.google.com/drive/folders/149Y5R6GIGDpBIaJhgG8rRaOslM21aA0Q?usp=sharing) unzip it.

- Download top 50 synonym file of counter-fitted-vectors from [here](https://drive.google.com/file/d/1AIz8Imvv8OmHxVwY5kx10iwKAUzD6ODx/view) and place it in the main directory.

- Download the glove 200 dimensional vectors from [here](https://nlp.stanford.edu/projects/glove/) unzip it.
 
#### How to Run:

Use the following command to get the results. 

For BERT model

```
python3 classification_attack.py \
        --dataset_path path_to_data_samples_to_attack  \
        --target_model Type_of_taget_model (bert,wordCNN,wordLSTM) \
        --counter_fitting_cos_sim_path path_to_top_50_synonym_file \
        --target_dataset dataset_to_attack (imdb,ag,yelp,yahoo,mr) \
        --target_model_path path_to_pretrained_target_model \
        --USE_cache_path " " \
        --max_seq_length 256 \
        --sim_score_window 40 \
        --nclasses classes_in_the_dataset_to_attack

```
Example of attacking BERT on IMDB dataset.

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

Example of attacking BERT on SNLI dataset. 

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
#### Results
The results will be available in **results_hard_label** directory for classification task and in **results_nli_hard_label** for entailment tasks.
For attacking other target models look at the ```commands``` folder.

#### Training target models
To train BERT on a particular dataset use the commands provided in the `BERT` directory. For training LSTM and CNN models run the `train_classifier.py --<model_name> --<dataset>`.

#### If you find our repository helpful, consider citing our work.
```
@article{maheshwary2020generating,
  title={Generating Natural Language Attacks in a Hard Label Black Box Setting},
  author={Maheshwary, Rishabh and Maheshwary, Saket and Pudi, Vikram},
  journal={arXiv preprint arXiv:2012.14956},
  year={2020}
}
```
