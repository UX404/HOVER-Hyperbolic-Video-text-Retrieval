DATA_PATH=./data/charades
python -m torch.distributed.launch --nproc_per_node=2 --master_port='28250' \
main_task_retrieval.py --do_train --num_thread_reader=0 \
--epochs=15 --batch_size=64 --n_display=50 \
--train_csv train_tree_deep \
--val_csv test_tree_deep \
--data_path ${DATA_PATH} \
--features_path ${DATA_PATH}/videos_compressed \
--output_dir ckpts/#name \
--lr 1e-4 --max_words 72 --max_frames 12 --batch_size_val 72 \
--datatype charades \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqLSTM \
--pretrained_clip_name ViT-B/32