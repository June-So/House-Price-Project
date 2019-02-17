import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def view_total_na(df,figsize=None,normalize=True):
    """ Visualisation des valeurs manquantes
        df : dataframe dont vous voulez visualiser les valeurs manquantes
        figsize : a tuple (width, height) in inches ; pour redimensier le plot
    """
    total_na = df.isna().sum().sort_values()
    if sum(total_na > 0) == 0:
        return "Aucune données manquantes :)"
    if normalize:
        total_na = round(total_na / df.shape[0], 2)
    col_miss_values = (total_na[total_na > 0]).index
    total_na[col_miss_values].plot(kind='barh',color='grey',figsize=figsize,title="Nombres de valeurs manquantes")
    plt.show()


def view_one_correlation(df,colname,figsize=None,ascending=False):
    """ Visualisation corrélation pour une colonne
        df: DataFrame
        col: colonne dont on veut connaitres les corrélations
        figsize : a tuple (width, height) in inches ; pour redimensier le plot
        ascending : bool
    """
    corr_price = df.corr()[[colname]].sort_values(colname,ascending=ascending)
    plt.figure(figsize=figsize)
    sns.heatmap(corr_price,annot=True)
    plt.show()

def view_one_categorical(df,x_colname,y_colname):
    """ Affiche en même temps la distribution et la corrélation d'une variable catégorique avec une variable quantitative """
    fig = plt.figure(figsize=(25,5))
    plt.subplot(1,4,1)
    df[x_colname].fillna('N/A').value_counts().plot(kind="bar",title=x_colname)
    plt.subplot(1,4,2)
    g = sns.boxplot(data=df,x=x_colname,y=y_colname)
    g.set_xticklabels(g.get_xticklabels(),rotation=90)
    plt.show()

def view_distributions(df,ncols=5):
    """ Affiche en une fois la distribution des colonnes d'un dataframe"""
    # Preparation de la grille d'affichage
    nrows = np.ceil(df.shape[1]/ncols)
    fig = plt.figure(figsize=(15,5*nrows))

    for count,col in enumerate(df.columns,1):
        # Distribution quantitative
        if df[col].dtype == object:
            plt.subplot(nrows,ncols,count)
            df[col].fillna('N/A').value_counts().plot(kind="bar",title=col)
        elif df[col].dtype in [np.float64,np.int64]:
            plt.subplot(nrows,ncols,count)
            df[col].plot(kind="hist",title=col)
    plt.show()

def view_correlations(df,y_colname,ncols=5,outliers=None):
    """ Pour afficher en une seule fois les corrélations de colonnes par rapport à une même variable en y"""
    # Preparation de la grille
    nrows = np.ceil(df.shape[1]/ncols)
    fig = plt.figure(figsize=(15,5*nrows))

    for count, col in enumerate(df.columns,1):
        # correlation x : quantitative y : qualitative
        if df[col].dtype == object:
            plt.subplot(nrows,ncols,count)
            g = sns.boxplot(data=df,x=col,y=y_colname)
            g.set_xticklabels(g.get_xticklabels(),rotation=90)
        # corrélation x : qualitative y : qualitative
        elif df[col].dtype in [np.float64,np.int64] :
            plt.subplot(nrows,ncols,count)
            plt.scatter(df[col],df[y_colname])
            if not outliers is None:
                is_outliers = df.index.isin(outliers)
                plt.scatter(df[is_outliers][col],df[is_outliers][y_colname])
            plt.title(col)
