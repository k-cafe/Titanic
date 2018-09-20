import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd

# Modeling Algorithms
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Modeling Helper
from sklearn.preprocessing import Imputer, Normalizer, scale
from sklearn.cross_validation import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFECV

# Visualisation
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns

# Configure visualisations
mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize'] = 8, 6
plt.interactive(False)


# Setup Helper functions
def plot_histograms(df, variables, n_rows, n_cols):
    fig = plt.figure(figsize=(16, 12))
    for i, var_name in enumerate(variables):
        ax = fig.add_subplot(n_rows, n_cols, i + 1)
        df[var_name].hist(bins=10, ax=ax)
        ax.set_title(
            'Skew: ' + str(round(float(df[var_name].skew()), )))  # + ' ' + var_name ) #var_name+" Distribution")
        ax.set_xticklabels([], visible=False)
        ax.set_yticklabels([], visible=False)
    fig.tight_layout()  # Improves appearance a bit.
    plt.show()


def plot_distribution(df, var, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, hue=target, aspect=4, row=row, col=col)
    facet.map(sns.kdeplot, var, shade=True)
    facet.set(xlim=(0, df[var].max()))
    facet.add_legend()


def plot_categories(df, cat, target, **kwargs):
    row = kwargs.get('row', None)
    col = kwargs.get('col', None)
    facet = sns.FacetGrid(df, row=row, col=col)
    facet.map(sns.barplot, cat, target)
    facet.add_legend()


def plot_correlation_map(df):
    corr = titanic.corr()
    _, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    _ = sns.heatmap(
        corr,
        cmap=cmap,
        square=True,
        cbar_kws={'shrink': .9},
        ax=ax,
        annot=True,
        annot_kws={'fontsize': 12}
    )


def describe_more(df):
    var = [];
    l = [];
    t = []
    for x in df:
        var.append(x)
        l.append(len(pd.value_counts(df[x])))
        t.append(df[x].dtypes)
    levels = pd.DataFrame({'Variable': var, 'Levels': l, 'Datatype': t})
    levels.sort_values(by='Levels', inplace=True)
    return levels


def plot_variable_importance(X, y):
    tree = DecisionTreeClassifier(random_state=99)
    tree.fit(X, y)
    plot_model_var_imp(tree, X, y)


def plot_model_var_imp(model, X, y):
    imp = pd.DataFrame(
        model.feature_importances_,
        columns=['Importance'],
        index=X.columns
    )
    imp = imp.sort_values(['Importance'], ascending=True)
    imp[: 10].plot(kind='barh')
    print(model.score(X, y))


# Load Data ---------------------------------------------------------------------------------------

# get DataSet from train / test CSV files
train = pd.read_csv('./models/train.csv')
test = pd.read_csv('./models/test.csv')

full = train.append(test, ignore_index=True)  # ignore_index: 行番号を振り直す
titanic = full[:891]

del train, test

print('DataSets: ', 'full: ', full.shape, 'titanic: ', titanic.shape)

# Run the code to see the variables, then read the variable description below to understand them
print(titanic.head())

# Anarytics, like min/max/mean/std ...etc
print(titanic.describe())

# データの相関関係を確認
plot_correlation_map(titanic)

# Plot distributions of Age of passangers who survived or did not survive
# ↓ 描画されたグラフの類似度が高ければ用いた変数は予測に適していない変数である
plot_distribution(titanic, var='Age', target='Survived', row='Sex')

# # Excersise 1 --------------------------------
# # Plot distributions of Fare of passangers who survived or did not survive
# plot_distribution(titanic, var='Fare', target='Survived')
#
# # Plot survival rate by Embarked
# # - 箱ひげチャート
# plot_categories(titanic, cat='Embarked', target='Survived')
#
# # Excersise 2 --------------------------------
# # Plot survival rate by Sex
# plot_categories(titanic, cat='Sex', target='Survived')
#
# # Excersise 3
# # Plot survival rate by Pclass
# plot_categories(titanic, cat='Pclass', target='Survived')
#
# # Excersise 4
# # Plot survival rate by SibSp
# plot_categories(titanic, cat='SibSp', target='Survived')
#
# # Excersise 5
# # Plot survival rate by Parch
# plot_categories(titanic, cat='Parch', target='Survived')

# 3. Data Preparation
# - データ加工：処理しやすいようにデータを加工する

# Transform Sex into binary values 0 and 1
# - 文字列を数値に変換
sex = pd.Series(np.where(full.Sex == 'male', 1, 0), name='Sex')

# Create new variable for every unique value of Embarked
# - 1/0に変換:　列が項目内に存在する種類（要素）。行がデータ数（データ番目）
embarked = pd.get_dummies(full.Embarked, prefix='Embarked')
print(embarked.head())

pclass = pd.get_dummies(full.Pclass, prefix='Pclass')
print(pclass.head())

# Fill missing values in variables

# データセットの設定
imputed = pd.DataFrame()  # データ格納用オブジェクト

# Fill missing values of Age with the average of Age (mean)
imputed['Age'] = full.Age.fillna(full.Age.mean())  # 平均値をNANの部分に挿入

# Fill missing values of Fare with the average of Fare (mean)
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())

print(imputed.head())

plt.show()
