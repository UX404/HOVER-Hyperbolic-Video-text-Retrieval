DATA_PATH=./data/msrvtt_data
python -m torch.distributed.launch --nproc_per_node=1 --master_port='22305' \
main_task_retrieval.py --do_eval --num_thread_reader=0 \
--epochs=5 --batch_size=64 --n_display=50 \
--train_csv ${DATA_PATH}/MSRVTT_train.9k.csv \
--val_csv ${DATA_PATH}/MSRVTT_JSFUSION_test.csv \
--data_path ${DATA_PATH}/MSRVTT_data.json \
--features_path ${DATA_PATH}/MSRVTT_Videos_Compressed \
--output_dir ckpts/#name \
--lr 1e-4 --max_words 72 --max_frames 12 --batch_size_val 48 \
--datatype msrvtt --expand_msrvtt_sentences  \
--feature_framerate 1 --coef_lr 1e-3 \
--freeze_layer_num 0  --slice_framepos 2 \
--loose_type --linear_patch 2d --sim_header seqLSTM \
--pretrained_clip_name ViT-B/32 \
--init_model ./ckpts/#name/pytorch_model.bin.0 \
--resume_model ./ckpts/#name/pytorch_opt.bin.0