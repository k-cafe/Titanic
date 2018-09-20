import pandas as pd
import numpy as np
# scikit-learnのインポートをします
from sklearn import tree

train = pd.read_csv("./models/train.csv")
test = pd.read_csv("./models/test.csv")

test_shape = test.shape
train_shape = train.shape
 
# print(test_shape)
# print(train_shape)

# ----------------
def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_row_columns = kesson_table.rename(columns = {0: '欠損数', 1: '%' })
    return kesson_table_row_columns

# print(kesson_table(train))
# print(kesson_table(test))

# データの欠損が多いので事前処理(データ加工) ---------------
train["Age"] = train["Age"].fillna(train["Age"].median())
train["Embarked"] = train["Embarked"].fillna("S")

print(kesson_table(train))

# 文字列データを数字に変換 --------------
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

print(train.head(10))


test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
test.Fare[152] = test.Fare.median()
 
print(test.head(10))

# trainの目的変数と説明変数を取得
target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]].values

my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit(features_one, target)

test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

my_prediction = my_tree_one.predict(test_features)

print(my_prediction.shape)
print(my_prediction)

# PassengerIdを取得
PassengerId = np.array(test["PassengerId"]).astype(int)
 
# my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])
 
# my_tree_one.csvとして書き出し
my_solution.to_csv("my_tree_one.csv", index_label = ["PassengerId"])