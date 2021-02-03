# Section 1: Data import, summary, pre-processing and visualisation (20%)
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

path = "clinical_dataset.xlsx"
dframe = pd.read_excel(path)
col_name = dframe.columns.values.tolist()  # 列名称
# # provide a summary of the dataset (e.g. mean values, standard deviations,  min/max values, etc. for each feature)
# print(dframe.describe())
# print("\n")
# print('what is the size of the data')
# print(dframe.shape)
# print("\n")
# print("How many features are there")
# print(len(col_name))
# print("\n")
# print("Are there any missing values")
# print(dframe.isnull().any())
# print("\n")
# print("Are there any missing values")
# print("yes,Status")
# print("\n")
# 数据做归一化
df = dframe[col_name[0:9]]
df = (df - df.min()) / (df.max() - df.min())
# # The first one shall be a box plot, which will include the two classes (“Status”),
# # i.e. healthy/cancerous, in the x-axis and the “Age” in the y-axis
# fig, axes = plt.subplots()
# dframe.boxplot(column='Age', by=['Status'], ax=axes)
#
# fig, axes = plt.subplots()
# ax = dframe['Age'][dframe['Status'] == "healthy"].plot.kde(ax=axes)
# ax1 = dframe['Age'][dframe['Status'] == "cancerous"].plot.kde(ax=axes)
# plt.legend(["healthy", "cancerous"])
# plt.show()




# Section 3: Designing algorithms (30%)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier

data = df.values
X = data[:, 0:9]
y = dframe['Status']
Y = []
for i in y:
    if (i == "healthy"):
        Y.append(0)
    else:
        Y.append(1)
Y = np.array(Y)


# 分割所需要的包
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

print("*************ANN**************")
model = MLPClassifier(hidden_layer_sizes=[500, 500], activation='logistic', solver='lbfgs', random_state=0, alpha=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
accuracy = model.score(X_test, y_test)
print(accuracy)
print(report)
#
# # random forests classifier
# print("*********random forests**********")
# from sklearn.ensemble import RandomForestClassifier
#
# model = RandomForestClassifier(n_estimators=1000, min_samples_split=5, bootstrap=True)
# model.fit(X_train, y_train)
# y_pred = model.predict(X_test)
# report = classification_report(y_test, y_pred)
# print(report)

#
#
# # Section 4: Model selection (20%)
# # 对于ANN，将每个隐藏层（记住有两个隐藏层）设置为50、500和1000个神经元。
# hidden_layer = [50, 500, 1000]
# # 对于随机森林，将树数设置为20、500和10000
# tree_num = [20, 500, 10000]
#
# from sklearn.model_selection import cross_val_score
#
# X = np.array(X)
# Y = np.array(Y)
# print("*************ANN 10 cross-validation**************")
# for i in hidden_layer:
#     print("hidden layer number: ", i)
#     model = MLPClassifier(hidden_layer_sizes=[i, i], activation='logistic', solver='lbfgs', random_state=0, alpha=0.1)
#     scores = cross_val_score(model, X, Y, cv=10)
#     print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# print("*********random forests 10 cross-validation**********")
# from sklearn.ensemble import RandomForestClassifier
#
# for i in tree_num:
#     print("tree number: ", i)
#     model = RandomForestClassifier(n_estimators=i, min_samples_split=5, bootstrap=True)
#     scores = cross_val_score(model, X, Y, cv=10)
#     print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
#
