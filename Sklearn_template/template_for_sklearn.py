## based on Titanic dataset
## data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

## visualization
#import seaborn as sns
#import matplotlib.pyplot as plt
# %matplotlib inline

## machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
import lightgbm as lgb


## data input
train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')

## preprocessing
## drop unnecessary columns
train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]
for dataset in combine:
    # regex
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
## mapping-categorical label
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

guess_ages = np.zeros((2,3))
for dataset in combine:
    for i in range(0, 2):
        for j in range(0, 3):
            guess_df = dataset[(dataset['Sex'] == i) & \
                                  (dataset['Pclass'] == j+1)]['Age'].dropna()

            # age_mean = guess_df.mean()
            # age_std = guess_df.std()
            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)

            age_guess = guess_df.median()

            # Convert random age float to nearest .5 age
            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

    for i in range(0, 2):
        for j in range(0, 3):
            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                    'Age'] = guess_ages[i,j]

    dataset['Age'] = dataset['Age'].astype(int)


_, bins = pd.cut(train_df['Age'], 5, retbins=True)

## bin cutoff
for dataset in combine:
    # dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
    # dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    # dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    # dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    # dataset.loc[ dataset['Age'] > 64, 'Age'] = 4
    dataset['Age'] = pd.cut(dataset['Age'],bins, labels=range(5))

combine = [train_df, test_df]
for dataset in combine:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

for dataset in combine:
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] <= 1, 'IsAlone'] = 1

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
combine = [train_df, test_df]

for dataset in combine:
    dataset['Age*Class'] = dataset.Age * dataset.Pclass
    dataset['Age*Class'] = dataset['Age*Class'].astype(int)

freq_port = train_df.Embarked.dropna().mode()[0]
for dataset in combine:
    dataset['Embarked'] = dataset['Embarked'].fillna(freq_port)
    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)
_, bins = pd.qcut(train_df['Fare'], 5, retbins=True)
bins[0] -=0.01
bins[-1] += 0.01

for dataset in combine:
    # dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    # dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    # dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    # dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    # dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Fare'] = pd.cut(dataset['Fare'], bins, labels=range(5)).astype(int)

combine = [train_df, test_df]

## oversample not necessary for all cases
train_df = train_df.sample(n=5*train_df.shape[0], replace=True, random_state = 42)
X_Train = train_df.drop("Survived", axis=1)
Y_Train = train_df["Survived"]
X_pred  = test_df.drop("PassengerId", axis=1).copy()

## OneHotEncoder
enc = OneHotEncoder()
enc.fit(X_Train)
X_Train = enc.transform(X_Train)
X_Pred = enc.transform(X_pred)

## Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_Train,Y_Train, test_size=0.2, random_state=42)

## estimator selection
estimator = RandomForestClassifier()
estimator.fit(X_train, y_train)
estimator.score(X_train, y_train)

estimator2 = BayesianGaussianMixture(n_components=2)
param_grid2 = {
               'weight_concentration_prior_type':['dirichlet_process', 'dirichlet_distribution'],
               'covariance_type' : ['full', 'tied', 'diag', 'spherical'],
              }
## parameter tuning
param_grid = {

    'n_estimators': [130,180,200],
    'criterion':['gini', 'entropy' ]
}
# estimator = lgb.LGBMClassifier(reg_alpha=0.01, reg_lambda=0.01)
# gbm = GridSearchCV(estimator, param_grid,cv=10)
gbm = GridSearchCV(estimator2, param_grid2,cv=10)
gbm.fit(X_train.toarray(), y_train)
print gbm.best_params_
print gbm.score(X_train.toarray(), y_train)

## prediction
pred = gbm.predict(X_Pred.toarray())

## classification_report
y_true, y_pred = y_test, gbm.predict(X_test.toarray())
print classification_report(y_true, y_pred)

## submission
submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": pred
    })
submission.to_csv('submission.csv', index=False)
