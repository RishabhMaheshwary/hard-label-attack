python3 attack_random_nli.py \
        --dataset_path data/snli  \
        --target_model infersent \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset snli \
        --target_model_path savedir/model.pickle \
        --USE_cache_path "nli_cache" \
        --sim_score_window 40 \
        --word_embeddings_path /scratch/glove.840B.300d.txt

