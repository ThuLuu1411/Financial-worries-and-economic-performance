#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 15:34:41 2022

@author: thuluu
"""

import numpy as np
import pandas as pd
from Preprocessing.Country import Country
from Preprocessing.Worry import Worry
from Preprocessing.GDP import GDP
from data import Finance

from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor 
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
import shap
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


import os


path = "/Users/vophuoctri/OneDrive/quaypha/data/"
os.chdir(path)


# al = [RandomForestRegressor(random_state=48),
#           XGBRegressor(random_state=0)]
al=[MLPRegressor(hidden_layer_sizes=(300,),
                      solver = "adam",
                      activation = "relu",
                      alpha=0.01,
                      learning_rate_init = 0.001,
                      tol=1e-4,
                      max_iter=7000,
                      early_stopping = False,
                      random_state=42)]
al_name = ["RandomForest_shapley2.csv"]#, "XGBoost_shapley2.csv", "MLP Neural Network_shapley2.csv"]
tree_base = ["RandomForest_shapley2.csv", "XGBoost_shapley2.csv"]


class Algorithm():
    def __init__(self, path, algorithm, algorithm_name, random_state, variable):
        self.variable = variable
        self.random_state = random_state
        self.path = path
        self.algorithm = algorithm
        self.algorithm_name = algorithm_name
        self.finance = Finance(path)
        self.data = self.finance.cf
        self.country_df = self.finance.country.df
        self.finance_series = self.finance.series
        self.y = self.clean_data()["gdp"]
        self.X = self.scale_data()[0]
        self.X_finance = self.scale_data()[1]
    def clean_data(self):
        df = self.data
        df = df[[x for x in df.columns if 'Most worrying' not in x]]
        df = df[~df['Region'].isnull()]
        df = df[~df['IncomeGroup'].isnull()]
        df = df[~df["gdp"].isnull()]
        # df = df[[x for x in df.columns if ('urban' not in x) and ('rural' not in x) and ('male' not in x) and ('primary' not in x) and ('secondary' not in x) and ('labor' not in x) and ('young' not in x) and ('older' not in x)]]
        df = df.dropna(axis=1, thresh=0.9999*len(df))
        df = df.dropna(axis=0, thresh=0.9999*len(df.columns))
        df = df[df['Country Name'].isin(['Sweden', 'Denmark', 'Norway', 'Finland', 'Cyprus', 'Lithuania'])==False]
        
        # #Loc du lieu chung
        # df = df[df['Worried about not having enough money for monthly expenses or bills: worried (% age 15+)']>0.2]
        # df = df[df['Worried about not having enough money for monthly expenses or bills: worried (% age 15+)']<0.9]
        # df = df[df['Worried about not being able to pay for medical costs in case of a serious illness or accident: worried (% age 15+)']>0.2]
        # df = df[df['Worried about not being able to pay for medical costs in case of a serious illness or accident: worried (% age 15+)']<0.9]
        # df = df[df['Worried about not having enough money for old age: worried (% age 15+)']>0.2]
        # df = df[df['Worried about not having enough money for old age: worried (% age 15+)']<0.9]
        # df = df[df['Worried about not being able to pay school fees or fees for education: worried (% age 15+)']>0.2]
        # df = df[df['Worried about not being able to pay school fees or fees for education: worried (% age 15+)']<0.9]

        # #Loc du lieu poorest
        # df = df[df['Worried about not having enough money for monthly expenses or bills: worried, income, poorest 40% (% age 15+)']>0.1]
        # df = df[df['Worried about not having enough money for monthly expenses or bills: worried, income, poorest 40% (% age 15+)']<0.90]
        # df = df[df['Worried about not being able to pay for medical costs in case of a serious illness or accident: worried, income, poorest 40% (% age 15+)']>0.1]
        # df = df[df['Worried about not being able to pay for medical costs in case of a serious illness or accident: worried, income, poorest 40% (% age 15+)']<0.90]
        # df = df[df['Worried about not having enough money for old age: worried, income, poorest 40% (% age 15+)']>0.1]
        # df = df[df['Worried about not having enough money for old age: worried, income, poorest 40% (% age 15+)']<0.90]
        # df = df[df['Worried about not being able to pay school fees or fees for education: worried, income, poorest 40% (% age 15+)']>0.1]
        # df = df[df['Worried about not being able to pay school fees or fees for education: worried, income, poorest 40% (% age 15+)']<0.90]

        # #Loc du lieu richest
        # df = df[df['Worried about not having enough money for monthly expenses or bills: worried, income, richest 60% (% age 15+)']>0.1]
        # df = df[df['Worried about not having enough money for monthly expenses or bills: worried, income, richest 60% (% age 15+)']<0.90]
        # df = df[df['Worried about not being able to pay for medical costs in case of a serious illness or accident: worried, income, richest 60% (% age 15+)']>0.1]
        # df = df[df['Worried about not being able to pay for medical costs in case of a serious illness or accident: worried, income, richest 60% (% age 15+)']<0.90]
        # df = df[df['Worried about not having enough money for old age: worried, income, richest 60% (% age 15+)']>0.1]
        # df = df[df['Worried about not having enough money for old age: worried, income, richest 60% (% age 15+)']<0.90]
        # df = df[df['Worried about not being able to pay school fees or fees for education: worried, income, richest 60% (% age 15+)']>0.1]
        # df = df[df['Worried about not being able to pay school fees or fees for education: worried, income, richest 60% (% age 15+)']<0.90]

        df = df.fillna(0).reset_index().drop('index', axis=1)
        return df
    def scale_data(self):
        X = self.clean_data().drop(["gdp", "Country Name", "Country Code", "Region", "IncomeGroup"], axis=1)
        X_sub = X[[
                    self.variable,
                    'Financial institution account (% age 15+)',
                    'Borrowed any money (% age 15+)',
                    'Used a mobile phone or the internet to pay bills (% age 15+)',
                    'Owns a debit or credit card (% age 15+)',
                   ]]
        # X_sub.columns = ['Expenses (worried)']#, 'Medical (worried)', 'Old age (not', 'Education (not)']
        #[[x for x in X.columns if 'Most worrying' not in x]]
        # worried = [x for x in X.columns if ': worried' not in x]
        # worried = [x for x in self.finance.worry.indicator_name if ('somewhat worried' not in x) and ('very worried' not in x)]
        # worried = [x for x in self.finance.worry.indicator_name if ('Most worrying' not in x) and ('Experience' not in x)]
        # worried = [x for x in worried if x in X.columns]
        # # print(worried)
        # X_sub[worried] = X[worried]
        robustscaler = RobustScaler()
        X_scaler = robustscaler.fit_transform(X_sub)
        X_df = pd.DataFrame(data=X_scaler, columns=X_sub.columns, index=X_sub.index)        
        return X_df, X_sub    
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=self.random_state)
        X_test = X_test.append(X_train.head(len(X_test)))
        y_test = y_test.append(y_train.head(len(y_test)))
        return self.X, X_test, self.y, y_test
    def fit(self):        
        model = self.algorithm
        model_fit = model.fit(self.split_data()[0].values, self.split_data()[2].values)        
        return model_fit
    def predict(self):
        model_fit = self.fit()
        return model_fit.predict(self.split_data()[1].values)
    def accuracy(self):
        return r2_score(self.split_data()[3], self.predict())
    def meaning(self):
        return r2_score(self.split_data()[2], self.fit().predict(self.split_data()[0].values))
    def mrse(self):
        return mean_squared_error(self.split_data()[2], self.fit().predict(self.split_data()[0].values))
    def mrse_out(self):
        return mean_squared_error(self.split_data()[3], self.predict())

# for i in al:
#     print(Algorithm(path, i).accuracy())

class SHAP(Algorithm):
    def __init__(self, path, algorithm, algorithm_name, random_state, variable):
        Algorithm.__init__(self, path, algorithm, algorithm_name, random_state, variable)
    def run_shap(self):
        if self.algorithm_name in tree_base:
            explainer = shap.TreeExplainer(self.fit(), self.X.values)
        else:
            explainer = shap.KernelExplainer(self.fit().predict, self.X.values)
        return explainer
    def shap_country(self, name_of_country):
        index = self.clean_data().iloc[:,:][self.clean_data()["Country Name"]==name_of_country].index
        shap_values = self.run_shap().shap_values(self.X.iloc[index,:], nsamples=500)
        p= shap.force_plot(self.run_shap().expected_value, shap_values, self.X.iloc[index,:], matplotlib = True, show = True)
        return(p)
    def shap_df(self):
        shap_values = self.run_shap().shap_values(self.X)
        shap_df =  pd.DataFrame(data = shap_values, index=self.clean_data()["Country Name"], columns=self.X.columns)
        shap_df.to_csv(self.algorithm_name)
        worried = self.X[[x for x in self.X.columns if ('Worried about' in x) and (('richest' in x) or ('poorest' in x) or (',' not in x))]]
        shap.summary_plot(shap_df[worried.columns.tolist()].values, worried, plot_type='violin')
        return shap_df

# l_random = []
# for i in range(100):
#     s = SHAP(path, al[0], 'MLP_shap2609.csv', random_state=i)
#     l_random.append([i, s.accuracy(), s.meaning()])
# random_df = pd.DataFrame(l_random)
   
worries_vars = [
                    'Worried about not having enough money for monthly expenses or bills: worried (% age 15+)',
                    'Worried about not being able to pay for medical costs in case of a serious illness or accident: worried (% age 15+)',
                    'Worried about not having enough money for old age: worried (% age 15+)',
                    'Worried about not being able to pay school fees or fees for education: worried (% age 15+)',
                    'Worried about not having enough money for monthly expenses or bills: worried, income, richest 60% (% age 15+)',
                    'Worried about not being able to pay for medical costs in case of a serious illness or accident: worried, income, richest 60% (% age 15+)',
                    'Worried about not having enough money for old age: worried, income, richest 60% (% age 15+)',
                    'Worried about not being able to pay school fees or fees for education: worried, income, richest 60% (% age 15+)',
                    'Worried about not having enough money for monthly expenses or bills: worried, income, poorest 40% (% age 15+)',
                    'Worried about not being able to pay for medical costs in case of a serious illness or accident: worried, income, poorest 40% (% age 15+)',
                    'Worried about not having enough money for old age: worried, income, poorest 40% (% age 15+)',
                    'Worried about not being able to pay school fees or fees for education: worried, income, poorest 40% (% age 15+)',

]

worries_ls = []
worries_models = []
worries_finance = []
for i in worries_vars:   
    s = SHAP(path, al[0], 'MLP_shap2609.csv', random_state=55, variable = i)
    worries_models.append(s.shap_df())
    worries_ls.append(i)
    

    
    
    
    
    
    
    
    
    
    
    
    
