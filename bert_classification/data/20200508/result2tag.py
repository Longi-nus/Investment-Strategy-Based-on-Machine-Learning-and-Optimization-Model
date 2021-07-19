import pandas as pd

df_class = pd.read_csv(r'./data/20200508/classes.txt', header=None)
df_prediction = pd.read_csv(r'./54_tag_focal_1000/test_res.csv', header=None, names=list(df_class[0]))

df_sec = pd.read_csv(r'./data/20200508/test_data.csv', header=None)
sentence_list = list(df_sec[0])

df_prediction['policy_point'] = sentence_list

cols = list(df_class[0])
df_prediction['max_prob'] = df_prediction[cols].max(1)
df_prediction['label'] = df_prediction[cols].idxmax(1)

df_prediction = df_prediction[['policy_point', 'label']]
df_origin = pd.read_csv(r'./text_data.csv',  dtype='object')

merge = pd.merge(df_origin, df_prediction, on=['policy_point'], how='inner')
merge.to_excel(r'./test_result.xlsx', index=None)