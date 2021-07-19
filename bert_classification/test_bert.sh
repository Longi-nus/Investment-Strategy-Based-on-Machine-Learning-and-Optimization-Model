# python testData_gen.py

BERT_BASE_DIR=./data/chinese_L-12_H-768_A-12
DATA_DIR=./data/20200508
MODEL_DIR=./54_tag_focal_1000
#export CUDA_VISIBLE_DEVICES=2

python multilabel_focal_loss.py \
 --task_name=multilabel	\
 --do_predict=true \
 --data_dir=$DATA_DIR/ \
 --vocab_file=$BERT_BASE_DIR/vocab.txt \
 --bert_config_file=$BERT_BASE_DIR/bert_config.json \
 --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
 --max_seq_length=256 \
 --train_batch_size=16 \
 --learning_rate=5e-5 \
 --num_train_epochs=200 \
 --output_dir=$MODEL_DIR \
 --res_name=test_res.csv
 
python data/20200508/result2tag.py