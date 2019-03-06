# Curso-Data-Science-Python-mlcourse.ai

Borrador personal de código

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

df.groupby('twp').size().sort_values(ascending=False).iloc[:5]

df['twp'].value_counts().sort_values(ascending=False).head(5)

df['title'].nunique()

df['title'].apply(lambda title: title.split(':')[0])

df['timeStamp'] = pd.to_datetime(df['timeStamp'])

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek) # hour, month..

dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

flights_df[(flights_df['ArrDelay']>0) & (flights_df['DepDelay']>0)].groupby(['UniqueCarrier'])\
  .agg({'ArrDelay': np.median,
        'DepDelay': np.median})\
  .sort_values(['DepDelay', 'ArrDelay'], ascending=True)\
  .iloc[0:10]
  
train['Age'].hist(bins=30,color='darkred',alpha=0.7)

sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')

sns.countplot(x='SibSp',data=train)

train.drop(['Sex','Embarked'],axis=1,inplace=True)

messages['message'].apply(len)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), train['Survived'], test_size=0.30, random_state=101)

from sklearn.linear_model import LogisticRegression #LinearRegression

logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test,predictions))

Regresión lineal:

plt.scatter(y_test,predictions)

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

SVM: 
from sklearn.svm import SVC

over partition by en python:

TempDF= DF.groupby(by=['ShopName'])['TotalCost'].sum()

TempDF= TempDF.reset_index() 

NewDF=pd.merge(DF , TempDF, how='inner', on='ShopName')

!pip install numpy==1.16.1

count_nonzero(A) / A.size
