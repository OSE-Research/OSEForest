#使用隔离森林（Isolation Forest）进行异常检测
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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


'''train data '''
white = load_data('white.csv', label=0)
data_white = feature_extract(white)
white_feature_domain = prehandle(data_white)
X_white = data_white.drop('DNSQueryName', axis=1)
# X_white = data_white.drop('DNSQueryName', axis=1, inplace=True)
X_white = pd.concat([white_feature_domain, X_white])
X_train_w = X_white.drop('label', axis=1)
print(X_train_w.info())
X_train_w = standard(X_train_w)


black = load_data('black.csv', label=0)
data_black = feature_extract(black)
black_feature_domain = prehandle(data_black)
X_black = data_black.drop('DNSQueryName', axis=1)
# X_black = data_black.drop('DNSQueryName', axis=1, inplace=True)
X_black = pd.concat([black_feature_domain, X_black])
X_train_b = X_black.drop('label', axis=1)
print(X_train_b.info())
X_train_b = standard(X_train_b)

model1 = IsolationForest()
model1.fit(X_train_w)

model2 = IsolationForest()
model2.fit(X_train_b)

#save model
joblib.dump(model1, './saved_model/if_w.pkl')
joblib.dump(model2, './saved_model/if_b.pkl')


'''test data'''
data_test = load_data(path='cdn.csv', label=0)
data_test = feature_extract(data_test)
feature_domain = prehandle(data_test)
data_test = data_test.drop('DNSQueryName', axis=1)
# data_test = data_test.drop('DNSQueryName', axis=1, inplace=True)
data_test = pd.concat([feature_domain, data_test])
y_test = data_test['label']
X_test = data_test.drop('label', axis=1)
print(X_test.describe())
print(X_test.info())
X_test = standard(X_test)



#load model
# model = joblib.load('./saved_model/xgb.pkl')

y_pred_train_w = model1.predict(X_train_w)
y_pred_test_w = model1.predict(X_test)
np.savetxt('./result/if_pred_w.txt', y_pred_test_w)
y_pred_train_b = model1.predict(X_train_b)
y_pred_test_b = model1.predict(X_test)
np.savetxt('./result/if_pred_b.txt', y_pred_test_b)


# 画图
xx, yy = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50)) # 生成网络数据
Z = model1.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest_white")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r) # 等高线

b1 = plt.scatter(X_train_w[:, 0], X_train_w[:, 1], c='white',
                 s=20, edgecolor='k')
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c='green',
                 s=20, edgecolor='k')

plt.axis('tight')
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend([b1, b2],
           ["training observations",
            "new cdn observations"],
           loc="upper left")
plt.show()