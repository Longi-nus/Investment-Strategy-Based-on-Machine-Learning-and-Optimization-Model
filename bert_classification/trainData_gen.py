import random
import pandas as pd
import sys


def add_tag(text):
    f = open('./data/20200508/classes.txt', 'r', encoding='utf-8')
    classes = f.readlines()
    f.close()
    classes = [a.strip() for a in classes]
    label_num = len(classes)

    for i in range(len(text)):
        single_label = ['0' for a in range(label_num)]
        non_empty = False
        for each_tag in text[i][1]:
            if each_tag not in classes:
                continue
            tag_index = classes.index(each_tag)
            single_label[tag_index] = '1'
            non_empty = True
        if not non_empty:
            single_label[-1] = '1'
        text[i] = '\t'.join([text[i][0]] + single_label)

    return text


def convert_xlsx_to_csv(Dir):
    res = []
    df = pd.read_excel(Dir)
    for index, row in df.iterrows():
        res.append([str(row['policy_point']).replace('\\n', '').replace('\n', '').replace('\t', '').replace(' ', ''), [str(row['label'])]])

    res = add_tag(res)

    # train_data = res[:int(0.8*len(res))]
    train_data = res
    val_data = res[int(0.8*len(res)):]
    random.shuffle(res)
    random.shuffle(val_data)

    f = open('./data/20200508/train_data.csv', 'w', encoding='utf-8')
    f.write('\n'.join(train_data))
    f.flush()
    f.close()
    f = open('./data/20200508/val_data.csv', 'w', encoding='utf-8')
    f.write('\n'.join(val_data))
    f.flush()
    f.close()

if __name__ == "__main__":
    convert_xlsx_to_csv(Dir=sys.argv[1])