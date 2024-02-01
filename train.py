#用于训练和测试机器学习模型的脚本，其中使用了XGBoost分类器来进行二分类任务（二分类标签为0和1，未知类别为2）
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, classification_report
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from file_extact import file_exact
from preprocess import *

FILENAME = 'train.csv'
TESTDIR = 'labeled_cdn'
TESTPATH = 'cdn.csv'
LABEL = 0
THRESHOLD = 0.9

'''train data '''
total = pd.read_csv('/projects/DNS_o/data/' + FILENAME, low_memory=False)
y = total['label']
X = total.drop('label', axis=1)


print(X.describe())
print(X.info())
X = standard(X)

# print("X[0]", X[0])
X_train = X
y_train = y
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.001, random_state=10)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)


'''test data'''
file_exact(dir=TESTDIR, filepath=TESTPATH)
data_test = load_data(path=TESTPATH, label=LABEL)
data_test = feature_extract(data_test)
feature_domain = prehandle(data_test)


data_test = pd.concat([feature_domain, data_test])
data_test.to_csv('cdn.csv', index=False, header=True)
data_test = data_test.drop('DNSQueryName', axis=1)
# data_test = data_test.drop('DNSQueryName', axis=1, inplace=True)

y_test = data_test['label']
X_test = data_test.drop('label', axis=1)



print(X_test.describe())
print(X_test.info())
X_test = standard(X_test)

'''train'''
# model = RandomForestRegressor(n_estimators=4)
# model = tree.DecisionTreeClassifier(criterion='gini', max_depth=None,
#     min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
#     max_features=None, random_state=None, max_leaf_nodes=None,
#     min_impurity_decrease=0.0, min_impurity_split=None,
#      class_weight=None, presort=False)

# model = neighbors.KNeighborsClassifier(n_neighbors=5, n_jobs=1)
model = XGBClassifier()

model.fit(X_train, y_train)
print("Train Accuracy: %lf" % model.score(X_train, y_train))

#save model
joblib.dump(model, './saved_model/xgb.pkl')


#load model
# model = joblib.load('./saved_model/xgb.pkl')

y_pred = model.predict_proba(X_test)
# y_pred = model.predict(X_test)

np.savetxt('./result/cdn_pred.txt', y_pred)

predictions = []
for value in y_pred:
    if value[0] >= THRESHOLD:
        predictions.append(0)
    elif value[1] >= THRESHOLD:
        predictions.append(1)

    else:
        predictions.append(2)

np.savetxt('./result/cdn.txt', predictions)
    # predictions = [value for value in y_pred]
white_num = predictions.count(0)
black_num = predictions.count(1)
unknown_num = predictions.count(2)

acc = white_num / len(predictions)
print("Test Acc:", acc)

print('预测正常条数：', white_num, '\n预测异常条数：', black_num, '\n预测未知条数：', unknown_num)


sk_pre = model.predict(X_test)
sk_pre = [round(value) for value in sk_pre]
sk_accuracy = accuracy_score(y_test, sk_pre)
print("Test Accuracy: %lf" % sk_accuracy)
report = classification_report(y_test, sk_pre, target_names=['0', '1'])
print(report)

r = classification_report(y_test, predictions, target_names=['0', '1', '2'])
print(r)
