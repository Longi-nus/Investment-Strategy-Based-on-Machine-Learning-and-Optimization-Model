import pandas as pd
import os
import time

file_list = os.listdir(r'./text_data')
file_list2= os.listdir(r'./text_data_pre')

for each_file in file_list:
    if each_file in file_list2:
        print(each_file)
        continue

    dframe = pd.read_csv('./text_data/' + each_file)['policy_point']
    dframe.to_csv(r'data/20200508/test_data.csv', index=None, header=None)
    os.system("python multilabel_focal_loss.py \
                --task_name=multilabel	\
                --do_predict=true \
                --data_dir=./data/20200508/ \
                --vocab_file=./data/chinese_L-12_H-768_A-12/vocab.txt \
                --bert_config_file=./data/chinese_L-12_H-768_A-12/bert_config.json \
                --init_checkpoint=./data/chinese_L-12_H-768_A-12/bert_model.ckpt \
                --max_seq_length=256 \
                --train_batch_size=16 \
                --learning_rate=5e-5 \
                --num_train_epochs=200 \
                --output_dir=./54_tag_focal_1000 \
                --res_name=test_res.csv" )
    
    # time.sleep(100)
    
    df_class = pd.read_csv(r'./data/20200508/classes.txt', header=None)
    df_prediction = pd.read_csv(r'./54_tag_focal_1000/test_res.csv', header=None, names=list(df_class[0]))

    print('==========================================={}==========================================='.format(each_file))
    print(len(df_prediction))
    print(df_prediction)

    df_sec = pd.read_csv(r'./data/20200508/test_data.csv', header=None)
    sentence_list = list(df_sec[0])

    df_prediction['policy_point'] = sentence_list

    cols = list(df_class[0])
    df_prediction['max_prob'] = df_prediction[cols].max(1)
    df_prediction['label'] = df_prediction[cols].idxmax(1)

    df_prediction = df_prediction[['policy_point', 'label']]

    df_prediction.to_csv('./text_data_pre/{}'.format(each_file), index=None)

file_list2= os.listdir(r'./text_data_pre')
df_pre = pd.DataFrame()
for each_file in file_list2:
    df_pre = df_pre.append(pd.read_csv('./text_data_pre/{}'.format(each_file)))


df_origin = pd.read_csv(r'./text_data.csv',  dtype='object')

merge = pd.merge(df_origin, df_pre, on=['policy_point'], how='inner')
merge.to_excel(r'./test_result.xlsx', index=None)