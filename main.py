import pandas as pd
import sklearn.model_selection as md
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

def get_complete_index_data():
    IndexData = pd.read_excel('Data/Raw_data/Index_Data.xlsx').set_index('查询码')
    human_result = pd.read_excel('Data/Raw_data/HumanId.xlsx')['Index ID'].tolist()
    IndexData.drop(IndexData.columns[:7], axis=1, inplace=True)
    IndexData.drop(IndexData.columns[-3:-1], axis=1, inplace=True)
    IndexData.rename(columns={'Y值': 'Y'}, inplace=True)
    IndexData['Y'] = IndexData.apply(lambda row: True if row['Y'] >= 2 else False, axis=1)
    IndexData['Y_human'] = IndexData.apply(lambda row: True if row.name in human_result else False, axis=1)
    IndexData.to_excel('Data/Processed_data/complete_index_data.xlsx')

# get full data
complete_index_data = pd.read_excel('Data/Processed_data/complete_index_data.xlsx').drop(columns='查询码')

# split to get train and test
X_train, X_test, y_train, y_test = md.train_test_split(complete_index_data.iloc[:, :-2],
                                                       complete_index_data.iloc[:, -2:],
                                                       test_size=0.33,
                                                       random_state=2)

train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)
# 只保留问题代号
data = [train_df, test_df]
for d in data:
    d.rename(columns=lambda x: x[:3], inplace=True)
    d.rename(columns={'请问您': 'Motivation',
                      '您是否': 'QRCode'}, inplace=True)

# 查看数据 数据大小 分析数据存在的问题
# 存在数据不平衡问题
# train: False 171, True 36 y_train.value_counts()
# test: False 80, True 22 y_test.value_counts()
# 97个特征 存在nan df.info()

# Data Preprocessing
# step 1 去掉一些不重要的特征[比如id,在哪里做的检测]/重复特征/太多空缺的/一个特征下类别太多的
for d in data:
    d.drop(columns=['B06', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12', 'C03', 'C04', 'D02', 'D05', 'D07', 'K02', 'K03',
                    'K05', 'B05', 'A06', 'QRCode'],
           inplace=True)
    d.drop(list(d.filter(regex='[E-J]')), axis=1, inplace=True)

# step 2 连续问题进行填充 & 去掉重复信息
# B01=1 B02=0, B06=0, B08=0, B10=1, & 去掉B01
# C01=1, C02=0, drop C01
# K01=1, K04=3, drop K01
# D01='从不', D03=0, D04=0, D06='从不' drop D01
for d in data:
    d.loc[d['B01'] == 1, ['B02']] = 0
    d.drop(columns=['B01'], inplace=True)
    d.loc[d['C01'] == 1, ['C02']] = 0
    d.drop(columns=['C01'], inplace=True)
    d.loc[d['K01'] == 1, ['K04']] = 3
    d.drop(columns=['K01'], inplace=True)
    d.loc[d['D01'] == '从不', ['D03', 'D04']] = 0
    d.loc[d['D01'] == '从不', ['D06']] = '从不'
    d.drop(columns=['D01'], inplace=True)

# step 3 convert to numeric
use_freq_dict = {'从未使用': 0, '偶尔使用（不到一半的情况）': 1, '经常使用（超过一半的情况）': 2, '每次都用': 3}
talk_freq_dict = {'从不': 0, '偶尔': 1, '经常': 2, '非常频繁': 3}
for d in data:
    d["B04"] = d.B04.map(use_freq_dict)
    d["D06"] = d.D06.map(talk_freq_dict)

# step 4 fill missing data
# K04 random
# Motivation random choose with same group
# B03 B04 fill with random
Motivation_in_each_group = train_df.groupby('K06').mean()['Motivation']
for d in data:
    d["K04"] = d["K04"].apply(
        lambda x: random.choice(d[-d["K04"].isna()]['K04'].values) if x != x else x)
    d["B03"] = d["B03"].apply(
        lambda x: random.choice([0, 1, 2]) if x != x else x)
    d["B04"] = d["B04"].apply(
        lambda x: random.choice([0, 1, 2, 3]) if x != x else x)
    d["Motivation"] = d.apply(lambda row: np.random.choice([0, 1], p=[1 - Motivation_in_each_group[row['K06']],
                                                                      Motivation_in_each_group[row['K06']]])
    if row['Motivation'] != row['Motivation'] else row['Motivation'], axis=1)

# step 5 create new features
# step 6 change data types
for d in data:
    d[['A01', 'A02', 'A03', 'A04', 'A05', 'B03', 'B04', 'D04', 'D06', 'K04', 'Motivation']] \
        = d[['A01', 'A02', 'A03', 'A04', 'A05', 'B03', 'B04', 'D04',
             'D06', 'K04', 'Motivation']].astype('category')

# step 7 Scaling the numerical data [对xgboost来说差别不大]
for d in data:
    stand_num_cols = d.select_dtypes(exclude=['bool', 'category'])
    d[stand_num_cols.columns] = StandardScaler().fit_transform(stand_num_cols)

X_train, y_train, y_train_human, X_test, y_test, y_test_human = train_df.iloc[:, :-2].copy(), \
                                                                train_df.iloc[:, -2].copy(), \
                                                                train_df.iloc[:, -1].copy(), \
                                                                test_df.iloc[:, :-2].copy(), \
                                                                test_df.iloc[:, -2].copy(), \
                                                                test_df.iloc[:, -1].copy(),
data = [X_train, X_test]
for d in data:
    d[['A01', 'A02', 'A03', 'A04', 'A05', 'B03', 'B04', 'D04', 'D06', 'K04', 'Motivation']] \
        = d[['A01', 'A02', 'A03', 'A04', 'A05', 'B03', 'B04', 'D04',
             'D06', 'K04', 'Motivation']].astype('int')

# --------------- unequal problem [显著提高了 recall] --------------
import imblearn
oversample = imblearn.over_sampling.SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)

# ---------- XGBoost --------------------------------------------------
from xgboost import XGBClassifier

xg = XGBClassifier(learning_rate=0.02, n_estimators=750,
                   max_depth=3, min_child_weight=1,
                   colsample_bytree=0.6, gamma=0.0,
                   reg_alpha=0.001, subsample=0.8, use_label_encoder=False, eval_metric='error')
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

print('------- XGB -------------------------------\n'
      'model_accuracy:{}, human_accuracy:{},\n'
      'model_recall:{}, human_recall:{},\n'
      'model_precision:{}, human_precision:{}, \n'
      'model_f1:{}, human_f1:{}, \n'.format(xg.score(X_test, y_test), accuracy_score(y_test, y_test_human),
                                            recall_score(y_test, y_pred), recall_score(y_test, y_test_human),
                                            precision_score(y_test, y_pred), precision_score(y_test, y_test_human),
                                            f1_score(y_test, y_pred), f1_score(y_test, y_test_human),))
# ----------------------------------------------------------------------

# ------------ RF ----------------------------------------------
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier(n_estimators=100)
RF.fit(X_train, y_train)
y_pred = RF.predict(X_test)
print('------- RF -------------------------------\n'
      'model_accuracy:{}, human_accuracy:{},\n'
      'model_recall:{}, human_recall:{},\n'
      'model_precision:{}, human_precision:{}, \n'
      'model_f1:{}, human_f1:{}, \n'.format(accuracy_score(y_test, y_pred), accuracy_score(y_test, y_test_human),
                                            recall_score(y_test, y_pred), recall_score(y_test, y_test_human),
                                            precision_score(y_test, y_pred), precision_score(y_test, y_test_human),
                                            f1_score(y_test, y_pred), f1_score(y_test, y_test_human),))
# ----------------------------------------------------------------------
# step 8 one-hot for category
X_train = pd.get_dummies(X_train, columns=['A01', 'A02', 'A03', 'A04', 'A05', 'B03', 'B04',
                                           'D04', 'D06', 'K04', 'Motivation'])

X_test = pd.get_dummies(X_test, columns=['A01', 'A02', 'A03', 'A04', 'A05', 'B03', 'B04',
                                         'D04', 'D06', 'K04', 'Motivation'])

missing_cols = set(X_train.columns) - set(X_test.columns)
for c in missing_cols:
    X_test[c] = 0
X_test = X_test[X_train.columns]

# ----------------- LR -------------------------------------------------
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression()
LR.fit(X_train, y_train)
y_pred = LR.predict(X_test)
print('------- LR -------------------------------\n'
      'model_accuracy:{}, human_accuracy:{},\n'
      'model_recall:{}, human_recall:{},\n'
      'model_precision:{}, human_precision:{}, \n'
      'model_f1:{}, human_f1:{}, \n'.format(accuracy_score(y_test, y_pred), accuracy_score(y_test, y_test_human),
                                            recall_score(y_test, y_pred), recall_score(y_test, y_test_human),
                                            precision_score(y_test, y_pred), precision_score(y_test, y_test_human),
                                            f1_score(y_test, y_pred), f1_score(y_test, y_test_human),))
# ----------------------------------------------------------------------


# ----------------- SVM -------------------------------------------------
from sklearn.svm import SVC
SVM = SVC()
SVM.fit(X_train, y_train)
y_pred = SVM.predict(X_test)
print('------- SVM -------------------------------\n'
      'model_accuracy:{}, human_accuracy:{},\n'
      'model_recall:{}, human_recall:{},\n'
      'model_precision:{}, human_precision:{}, \n'
      'model_f1:{}, human_f1:{}, \n'.format(accuracy_score(y_test, y_pred), accuracy_score(y_test, y_test_human),
                                            recall_score(y_test, y_pred), recall_score(y_test, y_test_human),
                                            precision_score(y_test, y_pred), precision_score(y_test, y_test_human),
                                            f1_score(y_test, y_pred), f1_score(y_test, y_test_human),))
# ----------------------------------------------------------------------


# ------------ Ensemble --------------------------------------
from sklearn.ensemble import VotingClassifier
votingC = VotingClassifier(estimators=[('LR', LR), ('SVM', SVM), ('XGB', xg), ("RandomForest", RF)],
                           n_jobs=4)
votingC.fit(X_train, y_train)
y_pred = votingC.predict(X_test)
print('------- votingC -------------------------------\n'
      'model_accuracy:{}, human_accuracy:{},\n'
      'model_recall:{}, human_recall:{},\n'
      'model_precision:{}, human_precision:{}, \n'
      'model_f1:{}, human_f1:{}, \n'.format(accuracy_score(y_test, y_pred), accuracy_score(y_test, y_test_human),
                                            recall_score(y_test, y_pred), recall_score(y_test, y_test_human),
                                            precision_score(y_test, y_pred), precision_score(y_test, y_test_human),
                                            f1_score(y_test, y_pred), f1_score(y_test, y_test_human),))
# ---------------------------------------------------------------------


# ------- XGB -------------------------------
# model_accuracy:0.8725490196078431, human_accuracy:0.7843137254901961,
# model_recall:0.7272727272727273, human_recall:0.36363636363636365,
# model_precision:0.6956521739130435, human_precision:0.5,
# model_f1:0.711111111111111, human_f1:0.4210526315789474,
# ------- RF -------------------------------
# model_accuracy:0.8725490196078431, human_accuracy:0.7843137254901961,
# model_recall:0.6818181818181818, human_recall:0.36363636363636365,
# model_precision:0.7142857142857143, human_precision:0.5,
# model_f1:0.6976744186046512, human_f1:0.4210526315789474,
# ------- LR -------------------------------
# model_accuracy:0.9019607843137255, human_accuracy:0.7843137254901961,
# model_recall:0.7727272727272727, human_recall:0.36363636363636365,
# model_precision:0.7727272727272727, human_precision:0.5,
# model_f1:0.7727272727272727, human_f1:0.4210526315789474,
# ------- SVM -------------------------------
# model_accuracy:0.8823529411764706, human_accuracy:0.7843137254901961,
# model_recall:0.6363636363636364, human_recall:0.36363636363636365,
# model_precision:0.7777777777777778, human_precision:0.5,
# model_f1:0.7000000000000001, human_f1:0.4210526315789474,
# ------- votingC -------------------------------
# model_accuracy:0.8823529411764706, human_accuracy:0.7843137254901961,
# model_recall:0.6818181818181818, human_recall:0.36363636363636365,
# model_precision:0.75, human_precision:0.5,
# model_f1:0.7142857142857143, human_f1:0.4210526315789474,
