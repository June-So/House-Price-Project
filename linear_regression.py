#!/usr/bin/env python
# coding: utf-8

# ## HOUSE DATA PRICE

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
import statsmodels.api as sm

from sklearn import linear_model, datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# ## Chargement des données

# In[2]:


train = pd.read_csv('D:/DevDataAI/Python/laurent Cours/Machine learning/Microsoft Azure/Projet/all/train.csv')
test = pd.read_csv('D:/DevDataAI/Python/laurent Cours/Machine learning/Microsoft Azure/Projet/all//test.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# Paramètres de visualisation pour jupyter

# In[5]:


plt.style.use('fivethirtyeight')
pd.options.display.max_columns = 100


# ## Exploration des données

# In[6]:


# enregistrer l'index et 
train_id = train['Id']
test_id = test['Id']

# suppression de l'Id
train.drop(columns ='Id', axis=1, inplace=True)
test.drop(columns ='Id', axis=1, inplace=True)

# sauvegarde du nombre de ligne du train et du test
n_train = train.shape[0]
n_test = test.shape[0]

# mettre le SalePrice le target dans une variable a part
y_train = train.SalePrice

# suppression de la colonne SalePrice
train.drop(['SalePrice'], axis=1, inplace=True)


# In[7]:


#concatenation des data train et test
df_inter = pd.concat((train, test)).reset_index(drop=True)


# ## Gestion des données manquantes
# Visualisation des données manquantes

# In[8]:


total_na = round((df_inter.isna().sum().sort_values())/df_inter.shape[0], 2)
col_miss_values = (total_na[total_na > 0]).index
total_na[col_miss_values].plot(kind='barh',color='grey',figsize=(8,5),title="Nombres de valeurs manquantes")


# On décide de supprimer les colonnes contenant plus de 15% de valeurs manquantes

# In[9]:


cols_low_na = total_na[total_na < 0.15]
df_inter = df_inter[cols_low_na.index]


# ## Gestion des valeurs manquantes
# - Que faire du GarageYrBlt ?
# - Bsmt => Vérifier qu'il  n'y a pas de basement
# 
# ### Question 1
# Les variables restantes sont-elles corrélées à notre prédiction ? -> Visualisation boxplot pour les qualitatives et scatterplot pour les quantitatives ( --> matrice de corrélation )<br>
# ### Question 2
# Est-ce que des variables nous apporte la même information ? -> Visualisation plot de corrélation ( --> matrice de corrélation )

# In[10]:


total_na = df_inter.isna().sum().sort_values()
total_na[total_na > 0].plot(kind='barh',color='grey',figsize=(8,5),title="Nombre de valeurs manquantes")


# ### ( A CONFIRMER )
# Le nombre de valeurs manquantes sont les mêmes pour les variables *Garage_*. Le manque d'informations provient de l'inexistence du Garage.
# <br><br>
# On décide de remplacer les n/a qualitatifs par un label "N/A"

# In[11]:


# cols_replace_na_cat = ["GarageCond","GarageType","GarageFinish","GarageQual"]
# df_train[cols_replace_na_cat] = df_train[cols_replace_na_cat].fillna('N/A')


# In[12]:


cols_replace_na_cat = ["GarageCond","GarageType","GarageFinish","GarageQual"]
for i in cols_replace_na_cat:
    df_inter[i].fillna('N/A', inplace=True)


# In[13]:


cols_replace_na_num =['GarageCars','GarageYrBlt','MasVnrArea','GarageArea']
for i in cols_replace_na_num:
    df_inter[i].fillna(0, inplace=True)


# ## Reconstitution de la data

# In[14]:


train_clean = df_inter[:n_train]
test_clean = df_inter[n_train:]

train_clean = pd.concat((train_clean, y_train), axis=1).reset_index(drop=True)
train_clean.head()


# ## Séléction de features
# Visualisations des corrélations

# In[15]:


sns.heatmap(train_clean.corr())


# In[16]:


corr_price = train_clean.corr()[['SalePrice']].sort_values('SalePrice',ascending=False)
plt.figure(figsize=(3,10))
sns.heatmap(corr_price,annot=True)


# On garde les variables quantitatives corrélées à plus de 25%

# In[17]:


corr_price = train_clean.corr()['SalePrice']
cols_features = corr_price[corr_price > 0.25].index
train_clean = train_clean[cols_features]


# ## Création de features

# Visualisations des corrélations

# In[18]:


sns.heatmap(train_clean.corr())


# In[19]:


corr_price = train_clean.corr()[['SalePrice']].sort_values('SalePrice',ascending=False)
plt.figure(figsize=(3,10))
sns.heatmap(corr_price,annot=True)


# On garde les variables quantitatives corrélées à plus de 25%

# In[20]:


corr_price = train_clean.corr()['SalePrice']
cols_features = corr_price[corr_price > 0.25].index
train_clean = train_clean[cols_features]
train_clean.head()


# In[21]:


train_clean.columns


# In[22]:


#Suppresion des colonnes 1rstFloor, 2ndFloor,GarageArea,
train_clean=train_clean.drop(['1stFlrSF', '2ndFlrSF','GarageArea','GarageYrBlt'],axis=1)
train_clean


# In[23]:


train_clean.columns


# In[24]:


train_clean.dtypes


# ## Transformations de valeurs

# In[25]:


train_clean['YearBuilt'].unique()


# In[26]:


#Transformation des colonnes 'YearBuilt','YearRemodAdd'
train_clean['YearBuilt'] = train_clean['YearBuilt'].apply(str)
train_clean['YearRemodAdd'] = train_clean['YearRemodAdd'].apply(str)
train_clean.head()


# In[27]:


#Transformation des float  en int
train_clean['GarageCars'] = train_clean['GarageCars'].apply(int)
train_clean['TotalBsmtSF'] = train_clean['TotalBsmtSF'].apply(int)
train_clean['BsmtFinSF1'] = train_clean['BsmtFinSF1'].apply(int)
train_clean['MasVnrArea'] = train_clean['MasVnrArea'].apply(int)


# In[31]:


#Transformation en catégorial

label_encoder = LabelEncoder()
train_clean['YearBuilt'] = label_encoder.fit_transform(train_clean['YearBuilt'])
train_clean['YearRemodAdd']=label_encoder.fit_transform(train_clean['YearRemodAdd'])
train_clean


# ## Gestion des outliers

# ## Modèle d'apprentissage
# - Définir un seuil final pour le test_size
# - On a temporairement retiré *GarageYrBlt* et *MasVnrArea* dû aux valeurs manquantes non traitées

# In[32]:


Y = train_clean['SalePrice']
X = train_clean.drop(['SalePrice'],axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3,random_state=100)
lm = LinearRegression()
lm.fit(X_train,y_train)
lm.predict(X_test)
lm.score(X_test,y_test)

