 python3 nli_attack.py \
        --dataset_path data/snli  \
        --target_model esim \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset snli \
        --target_model_path esim_pretrained/snli.pth.tar \
        --USE_cache_path "nli_cache" \
        --sim_score_window 40 \
        --word_embeddings_path esim_pretrained/snli.pkl && 
 python3 nli_attack.py \
        --dataset_path data/mnli_matched  \
        --target_model esim \
        --counter_fitting_cos_sim_path mat.txt \
        --target_dataset mnli \
        --target_model_path esim_pretrained/mnli.pth.tar \
        --USE_cache_path "nli_cache" \
        --sim_score_window 40 \
        --word_embeddings_path esim_pretrained/mnli.pkl