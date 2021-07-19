import pandas as pd
eval_result = pd.read_csv(r'./54_tag_focal_1000/eval_results.txt', header = None, sep = '=')
eval_result['is_accuracy'] = eval_result[0].str.startswith('accuracy')
eval_result['is_precision'] = eval_result[0].str.startswith('precision')
eval_result['is_recall'] = eval_result[0].str.startswith('recall')

precision_mean = float(pd.DataFrame(eval_result.groupby('is_precision')[1].mean()).loc[True])
recall_mean = float(pd.DataFrame(eval_result.groupby('is_recall')[1].mean()).loc[True])
f1_score = (2*precision_mean*recall_mean) / (precision_mean + recall_mean)

with open(r'./54_tag_focal_1000/eval_results.txt', 'a') as f:
    f.write('\nf1_score: {:.2f}'.format(f1_score))
