# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 16:33:12 2022

@author: amrit
"""

# In this code different Classification algorithms have been used to classify the Income IItm dataset.

from numpy import mean
from numpy import std
from pandas import read_csv
from matplotlib import pyplot
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier

# load the dataset
def load_dataset(full_path):
	# load the dataset as a numpy array
	dataframe = read_csv(full_path, header=None, na_values='?')
	# drop rows with missing
	dataframe = dataframe.dropna()
	# split into inputs and outputs
	last_ix = len(dataframe.columns) - 1
	X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
	# select categorical and numerical features
	cat_ix = X.select_dtypes(include=['object', 'bool']).columns
	num_ix = X.select_dtypes(include=['int64', 'float64']).columns
	# label encode the target variable to have the classes 0 and 1
	y = LabelEncoder().fit_transform(y)
	return X.values, y, cat_ix, num_ix

# evaluate a model
def evaluate_model(X, y, model):
	# define evaluation procedure
	cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
	# evaluate model
	scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
	return scores

# define models to test
def get_models():
	models, names = list(), list()
	# CART
	models.append(DecisionTreeClassifier())
	names.append('CART')
	# SVM
	models.append(SVC(gamma='scale'))
	names.append('SVM')
	# Bagging
	models.append(BaggingClassifier(n_estimators=100))
	names.append('BAG')
	# RF
	models.append(RandomForestClassifier(n_estimators=100))
	names.append('RF')
	# GBM
	models.append(GradientBoostingClassifier(n_estimators=100))
	names.append('GBM')
	return models, names

# define the location of the dataset
full_path = 'E:\\AmritDocuments\\pythonfiles\\examples\\logistic( income) IITM data  EDA.csv'
# load the dataset
X, y, cat_ix, num_ix = load_dataset(full_path)
# define models
models, names = get_models()
results = list()
# evaluate each model
for i in range(len(models)):
	# define steps
	steps = [('c',OneHotEncoder(handle_unknown='ignore'),cat_ix), ('n',MinMaxScaler(),num_ix)]
	# one hot encode categorical, normalize numerical
	ct = ColumnTransformer(steps)
	# wrap the model i a pipeline
	pipeline = Pipeline(steps=[('t',ct),('m',models[i])])
	# evaluate the model and store results
	scores = evaluate_model(X, y, pipeline)
	results.append(scores)
	# summarize performance
	print('>%s %.3f (%.3f)' % (names[i], mean(scores), std(scores)))
# plot the results
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()



#-------------------------------Answer 2 ---------------------------------------------------
import pandas as pd
import numpy as np
import datetime
import seaborn as sns
import matplotlib.pyplot as plt




df= pd.read_csv("E:\\TRENT\\SEM 3\\BIG DATA\\Assignments\\archive2\\OnlineRetail.csv",header = 0 )

df.info()

df.isna().sum()

df.dropna(inplace=True)

# Feature generation

df["Amount"] = df["Quantity"]*df["UnitPrice"]

df_amount = df.groupby(['CustomerID'],as_index=False)["Amount"].sum()
df_amount.head()

# UDF to convert string to datetime
def convertDate(x):
    conv_date = datetime.datetime.strptime(x, '%d-%m-%Y %H:%M')
    return conv_date

def splitDate(y):
    y = str(y)
    num_days = y.split()[0]
    num_days = int(num_days)
    return num_days

df['InvoiceDate'] = df.loc[:,'InvoiceDate'].apply(convertDate)

max_date = df['InvoiceDate'].max()


df['Recency'] = max_date - df['InvoiceDate']

df['Recency'] = df['Recency'].apply(splitDate)


df_rec_purch = df.groupby(['CustomerID'],as_index=False)["Recency"].min()
df_rec_purch.head()


# Attribute : Frequency of purchase
df_purchase_freq = df.groupby(['CustomerID'],as_index=False)["InvoiceNo"].count()
df_purchase_freq.rename(columns={'CustomerID':'CustomerID','InvoiceNo':'Frequency'},inplace=True)
df_purchase_freq.head()


newdf = df_amount.merge(df_rec_purch,how='left',on='CustomerID')
newdf = newdf.merge(df_purchase_freq,how='left',on="CustomerID")


def viewDist(df):
    sns.boxplot(data=df)
    plt.xticks(rotation=90)
    plt.show
    
df_out = newdf.iloc[:,1:4]
viewDist(df_out)

df_out.info()

def remove_outlier_IQR(df):
    Q1=df.quantile(0.25)
    Q3=df.quantile(0.75)
    IQR=Q3-Q1
    df_final=df[~((df<(Q1-1.5*IQR)) | (df>(Q3+1.5*IQR)))]
    return df_final

df_iqr = remove_outlier_IQR(df_out)    

df_iqr.info()

df_iqr.fillna(0,inplace=True)
viewDist(df_iqr)

from sklearn.preprocessing import StandardScaler
# define standard scaler
scaler = StandardScaler()
# transform data
df_scalar = scaler.fit_transform(df_iqr)

df_scalar1 = pd.DataFrame(df_scalar)
df_scalar1.columns = ['Amount' , 'Frequency' , 'Recency']
df_scalar1.head()

from sklearn.cluster import KMeans

# Building a model with 3 clusters
kmeans= KMeans(n_clusters=3)
kmeans.fit(df_iqr[['Amount', 'Frequency', 'Recency']])
pred = kmeans.predict(df_iqr[['Amount', 'Frequency', 'Recency']])


kmeans.cluster_centers_


finaldf = df_iqr.join(pd.DataFrame(pred,columns=["pred"]))

# Visualizing Results
fig, ax =plt.subplots(nrows= 1, ncols = 3, figsize= (14,6))
ty=sns.stripplot(x='pred', y='Amount', data=finaldf, s=8, ax = ax[0], palette='magma_r')
sns.despine(left=True)
ty.set_title('Clusters based on different Amounts')
ty.set_ylabel('Total Spent')
ty.set_xlabel('Clusters')

tt=sns.boxplot(x='pred', y='Frequency', data=finaldf, ax = ax[1], palette='coolwarm_r')
tt.set_title('Clusters based on Number of Transactions')
tt.set_ylabel('Total Transactions')
tt.set_xlabel('Clusters')

tr=sns.boxplot(x='pred', y='Recency', data=finaldf, ax = ax[2], palette='magma_r')
tr.set_title('Clusters based on Last Transaction')
tr.set_ylabel('Last Transactions (Days ago)')
tr.set_xlabel('Clusters')


# Building a model with 6 clusters
kmeans= KMeans(n_clusters=6)
kmeans.fit(df_iqr[['Amount', 'Frequency', 'Recency']])
pred = kmeans.predict(df_iqr[['Amount', 'Frequency', 'Recency']])


kmeans.cluster_centers_


finaldf = df_iqr.join(pd.DataFrame(pred,columns=["pred"]))

# Visualizing Results
fig, ax =plt.subplots(nrows= 1, ncols = 3, figsize= (14,6))
ty=sns.stripplot(x='pred', y='Amount', data=finaldf, s=8, ax = ax[0], palette='magma_r')
sns.despine(left=True)
ty.set_title('Clusters based on different Amounts')
ty.set_ylabel('Total Spent')
ty.set_xlabel('Clusters')

tt=sns.boxplot(x='pred', y='Frequency', data=finaldf, ax = ax[1], palette='coolwarm_r')
tt.set_title('Clusters based on Number of Transactions')
tt.set_ylabel('Total Transactions')
tt.set_xlabel('Clusters')

tr=sns.boxplot(x='pred', y='Recency', data=finaldf, ax = ax[2], palette='magma_r')
tr.set_title('Clusters based on Last Transaction')
tr.set_ylabel('Last Transactions (Days ago)')
tr.set_xlabel('Clusters')













