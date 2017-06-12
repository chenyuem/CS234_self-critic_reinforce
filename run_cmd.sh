
prepro_feature() {
    (
    python scripts/prepro_feats_npy.py --input_json ./data/dataset_coco.json --output_dir /datadrive/resnet_features/cocotalk --images_root /datadrive/models/im2txt/data/mscoco/raw-data/ --endding $1
    )
}

train_fc() {
    (
    python -u train.py --input_json ./data/cocotalk.json --input_fc_dir ./data/cocotalk_fc --input_att_dir ./data/cocotalk_att --input_label_h5 ./data/cocotalk_label.h5 --id fc --caption_model fc --beam_size 1 --learning_rate 5e-4 --learning_rate_decay_start 0 --scheduled_sampling_start 0 --save_checkpoint_every 6000 --val_images_use 5000 --checkpoint_path log_fc_xe | tee $1
    # --language_eval 1
    )
}

eval_fc() {
    (
    python -u eval.py --dump_images 0 --num_images 5000 --model ./log_fc/model-best.pth --language_eval 1 --infos_path ./log_fc/infos_fc-best.pkl | tee $1
    )
}

eval_fc_xe() {
    (
    python -u eval.py --dump_images 0 --num_images 5000 --model ./log_fc_xe/model-best.pth --language_eval 0 --infos_path ./log_fc_xe/infos_fc-best.pkl | tee $1
    )
}

train_fc_rl() {
    (
    python -u train_rl.py --caption_model fc --rnn_size 512 --batch_size 10 --seq_per_img 5 --input_encoding_size 512 --train_only 0 --id fc_rl --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --beam_size 1 --learning_rate 5e-5 --optim adam --optim_alpha 0.9 --optim_beta 0.999 --checkpoint_path log_fc_rl --start_from log_fc_rl --save_checkpoint_every 5000 --language_eval 1 --val_images_use 5000 | tee $1
    )
}

train_att2in_rl() {
    (
    python -u train_rl.py --caption_model att2in --rnn_size 512 --batch_size 10 --seq_per_img 5 --input_encoding_size 512 --train_only 0 --id att2in_rl --input_json data/cocotalk.json --input_fc_dir data/cocotalk_fc --input_att_dir data/cocotalk_att --input_label_h5 data/cocotalk_label.h5 --beam_size 1 --learning_rate 5e-5 --optim adam --optim_alpha 0.9 --optim_beta 0.999 --checkpoint_path log_att2in_rl --start_from log_att2in_rl --save_checkpoint_every 5000 --language_eval 1 --val_images_use 5000 | tee log-train-att2in-rl.txt
    )
}

eval_fc_rl() {
    (
    python -u eval.py --dump_images 0 --num_images 5000 --model ./log_fc_rl/model-best.pth --language_eval 1 --infos_path ./log_fc_rl/infos_fc_rl-best.pkl | tee $1
    )
}

eval_att2in() {
    (
    python eval.py --dump_images 0 --num_images 5000 --model ./log_att2in/model-best.pth --language_eval 1 --infos_path ./log_att2in/infos_att2in-best.pkl
    )
}

eval_att2in_rl() {
    (
    python eval.py --dump_images 0 --num_images 5000 --model ./log_att2in_rl/model-best.pth --language_eval 1 --infos_path ./log_att2in_rl/infos_att2in_rl-best.pkl
    )
}

eval_att2in_rl_beam_search() {
    (
    python eval.py --dump_images 0 --num_images 5000 --beam_size 1 --model ./log_att2in_rl/model-best.pth --language_eval 1 --infos_path ./log_att2in_rl/infos_att2in_rl-best.pkl
    )
}

visualize_att() {
    (
    cat new_test_images_list.txt | parallel -j2 'python demo_eval.py --demo_image {}'
    )
}

live_demo_given_url() {
    (
    python demo_eval.py --demo_image $1
    )
}
