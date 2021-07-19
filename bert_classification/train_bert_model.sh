cd 54_tag_focal_1000
rm -rf *
cd ..

python trainData_gen.py 'train_data.xlsx'

sh train_bert.sh

python cal_f1_score.py