#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 19:03:58 2022

@author: vophuoctri
"""

import numpy as np
import pandas as pd
from Preprocessing.Country import Country
from Preprocessing.Worry import Worry
from Preprocessing.GDP import GDP
from data import SHAP_DF
from data import Finance
from Algorithm import SHAP
from sklearn.neural_network import MLPRegressor
import random

import shap
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from shap import Explanation
from shap.plots._waterfall import waterfall


path = '/Users/vophuoctri/OneDrive/quaypha/data/'
file_name = 'MLP Neural Network_shapley2.csv'

al=[MLPRegressor(hidden_layer_sizes=(100,),
                     solver = "adam",
                     activation = "relu",
                     #learning_rate_init = 0.001,
                     tol=1e-4,
                     max_iter=7000,
                     early_stopping = False,
                     random_state=30)]

shap_ = SHAP_DF(path, file_name)
finance = Finance(path)

shap_df = shap_.country_shap()
finance_df = finance.country_finance()[shap_df.columns.tolist()]
finance_df = finance_df[finance_df["Country Name"].isin(shap_df["Country Name"].unique().tolist())]
shap_df = shap_df.set_index(['Country Name', 'Country Code', 'Region', 'IncomeGroup'])
finance_df = finance_df.set_index(['Country Name', 'Country Code', 'Region', 'IncomeGroup'])

#Summary plot
plt.figure(figsize=(10,10))
shap.summary_plot(shap_df.values, finance_df, show=False)
plt.title(file_name)
plt.show()

#Summary plot for worry
worry_indicator = Worry(path).indicator_name
worry_indicator = list(set(worry_indicator) & set(shap_df.columns.tolist()))
plt.figure(figsize=(10,10))
shap.summary_plot(shap_df[worry_indicator].values, finance_df[worry_indicator])
plt.show()

#Feature Importances all
fi = np.abs(shap_df).mean().sort_values()
plt.figure(figsize=(10,10))
fi.tail(20).plot.barh()
plt.show()

#Feature Importances for worry
plt.figure(figsize=(10,10))
fi[worry_indicator].sort_values().tail(20).plot.barh()
plt.show()

#Combining group of worry
dfT = shap_df.T.reset_index()
dfT = dfT.rename(columns={"index": "Indicator Name"})
dfT = dfT.merge(Worry(path).df, on="Indicator Name")

ls_worry = ['Series',
            'maingroup',
             'subgroup',
                  'how',
               'branch']


branches = [['female',  'male'], 
            ['young', 'older'],
            ['primary', 'secondary'],
            ['richest', 'poorest'],
            ['in.*labor', 'out.*labor'],
            ['urban', 'rural']
            ]

maingroup = Worry(path).maingroup
subgroup = Worry(path).subgroup

highincome = shap_df.reset_index()[shap_df.reset_index()["IncomeGroup"]=='High income']["Country Name"].tolist()
lowincome = shap_df.reset_index()[shap_df.reset_index()["IncomeGroup"]=='Low income']['Country Name'].tolist()
lowermiddleincome = shap_df.reset_index()[shap_df.reset_index()["IncomeGroup"]=='Lower middle income']['Country Name'].tolist()
uppermiddleincome = shap_df.reset_index()[shap_df.reset_index()["IncomeGroup"]=='Upper middle income']['Country Name'].tolist()


df = dfT[dfT["maingroup"]=="Worried "]
df_group = df.groupby(["how", "branch"]).sum()
df_group = df_group.reset_index()

for i,j in enumerate(random.sample(highincome, 6),1):
    plt.figure(figsize=(20,10))
    for x in df.columns:
        if j in x:
            col = x
    plt.subplot(2,3,i, title=j)
    sns.barplot(x=col, y="how", hue="branch", data=df_group)
    plt.xlabel("shapley value")
    # plt.legend("")
    # plt.show()
# plt.legend(#title = columns_bar,
#            #handles=handles, 
#            bbox_to_anchor=(1.05, -0.1), ncol=2)

plt.show()


# 
for i in maingroup:
    lsa = [x.replace(i,"") for x in shap_df.columns.tolist() if re.search("^"+i, x)]
    lsb = [x for x in shap_df.columns.tolist() if re.search("^"+i, x)]
    for j in branches:
        ls1a = [x.replace(": ","") for x in lsa if (re.search(j[0], x)) or (re.search(j[1],x))]
        ls1b = [x for x in lsb if (re.search(j[0], x)) or (re.search(j[1],x))]
        if i == maingroup[0]:
            for g in subgroup:
                try:
                    g = 'about' + g 
                    ls2a = [x.replace(g, "") for x in ls1a if g in x]
                    ls2b = [x for x in ls1b if g in x]
                    finance_df2 = finance_df[ls2b]
                    finance_df2.columns = ls2a
                    shap.summary_plot(shap_df[ls2b].values, finance_df2, show=False)
                    plt.title(i+'-'+g+':'+j[0]+" vs "+j[1])
                    plt.show()
                except:
                    continue
        else:
            finance_df2 = finance_df[ls1b]
            finance_df2.columns = ls1a
            shap.summary_plot(shap_df[ls1b].values, finance_df2, show=False)
            plt.title(i+":"+j[0]+" vs "+j[1])
            plt.show()
        
        

shap_ = SHAP(path, al[0], 'MLP')
exp = Explanation(shap_df[worry_indicator].values, 
                  np.full((len(shap_df)), shap_.predict().mean()), 
                  data=shap_.X[worry_indicator].values, 
                  worry = shap_df[worry_indicator].values,
                  feature_names=shap_.X[worry_indicator].columns)

country = shap_df.reset_index()['Country Name']
        
for i in range(len(shap_df)):
    idx = i
    waterfall(exp[idx], show=False)
    plt.title(country.iloc[i])
    plt.show()
        













