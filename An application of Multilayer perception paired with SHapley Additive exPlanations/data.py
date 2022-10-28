#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 00:08:43 2022

@author: thuluu
"""

import numpy as np
import pandas as pd
from Preprocessing.Country import Country
from Preprocessing.Worry import Worry
from Preprocessing.GDP import GDP

file_name = "Databank-wide.xlsx"
sheet_name = "Data"
sheet_name2 = "Series_Table"


class Finance():
    def __init__(self, path):
        self.path = path
        self.country = Country(path)
        self.worry = Worry(path)
        self.gdp = GDP(path)
        self.df = self.get_data_final()
        self.cf = self.country_finance()
        self.series = self.series()
    def get_data(self):
        df = pd.read_excel(self.path+file_name, sheet_name = sheet_name, header=1)
        df = df.rename(columns={df.columns.tolist()[0]: "Country Name"})
        df = df[df[df.columns[3]]==2021]
        for i in df.columns.tolist():
            if "Unname" in i:
                df = df.drop(i, axis=1)
        return df
    def get_data_final(self):
        df = self.get_data()
        subgroup = self.worry.subgroup
        main_df = df[[x for x in df.columns if 'Worried about' in x]]
        for i in subgroup[:4]:
            sub_df = main_df[[x for x in main_df.columns if i in x]]
            for j in ['female',  ' male', 
                    'young', 'older',
                    'primary', 'secondary',
                    'income, richest 60%', 'income, poorest 40%',
                    'in labor', 'out of labor',
                    'urban', 'rural',
                    ]:
                branch_df = sub_df[[x for x in sub_df.columns if j in x]]
                # print(1 - branch_df[[x for x in branch_df.columns if 'not worried at all' in x]])
                # main_df['Worried about'+i+': worried, '+j+' (% age 15+)'] = branch_df[[x for x in branch_df.columns if ('very worried' in x) or ('somewhat worried' in x)]].sum(axis=1)
                main_df['Worried about'+i+': worried, '+j+' (% age 15+)'] = 1 - branch_df[[x for x in branch_df.columns if 'not worried at all' in x]]

        for i in subgroup[:4]:
            sub_df = main_df[[x for x in main_df.columns if i in x]]
            branch_df = sub_df[[x for x in sub_df.columns if ',' not in x]]
            # main_df['Worried about'+i+': worried (% age 15+)'] = branch_df[[x for x in branch_df.columns if ('very worried' in x) or ('somewhat worried' in x)]].sum(axis=1)
            main_df['Worried about'+i+': worried (% age 15+)'] = 1 - branch_df[[x for x in branch_df.columns if 'not worried at all' in x]]
        worried_df = main_df[[x for x in main_df.columns if ('not worried al all' not in x) and ('very worried' not in x) and ('somewhat worried' not in x)]]
        df[worried_df.columns] = worried_df
        df = df.drop([x for x in main_df.columns if ('very worried' in x) or ('somewhat worried' in x)], axis=1)
        return df   
    def series(self):
        series = pd.read_excel(self.path+file_name, sheet_name = sheet_name2)
        columns = pd.DataFrame(data = self.df.columns[1:], columns=["Indicator Name"])
        series = series[["Series", "Indicator Name"]].merge(columns, on="Indicator Name")
        sr = {}
        for i in series["Indicator Name"].unique().tolist():
            sr[i] = series[series["Indicator Name"]==i]["Series"].values.tolist()[0]
        return sr
    def country_finance(self):
        country_df = self.country.df
        finance_df = self.df
        gdp_df = self.gdp.df
        cf_df = country_df.merge(gdp_df, on="Country Name", how="left")
        cf_df = cf_df.merge(finance_df, on="Country Name", how="right")
        return cf_df
    def country_worry(self, group_country, worry_issue):
        cr = self.country_finance()
        wi = self.worry.df[worry_issue].drop_duplicates()
        cols = ["Country Name", group_country] + list(set(wi["Indicator Name"].values.tolist()) & set(cr.columns.tolist())) 
        cr = cr[cols]
        cr = cr.melt(id_vars=["Country Name", group_country], value_vars=cols[2:], var_name="Indicator Name")
        return cr.merge(wi, on="Indicator Name")
    
class SHAP_DF():
    def __init__(self, path, algorithm):
        self.filename = algorithm
        self.path = path
        self.sheet_name = sheet_name
        self.country = Country(path)
        self.worry = Worry(path)
        self.gdp = GDP(path)
        self.df = self.get_data()
        self.cs = self.country_shap()
        self.series = self.series()
    def get_data(self):
        df = pd.read_csv(self.path+self.filename, header=0)
        return df
    def series(self):
        series = pd.read_excel(self.path+file_name, sheet_name = sheet_name2)
        columns = pd.DataFrame(data = self.df.columns[1:], columns=["Indicator Name"])
        series = series[["Series", "Indicator Name"]].merge(columns, on="Indicator Name")
        sr = {}
        for i in series["Indicator Name"].unique().tolist():
            sr[i] = series[series["Indicator Name"]==i]["Series"].values.tolist()[0]
        return sr
    def country_shap(self):
        country_df = self.country.df
        shap_df = self.df
        cs_df = country_df.merge(shap_df, on="Country Name", how="right")
        return cs_df
    def country_worry(self, group_country, worry_issue):
        cr = self.country_shap()
        wi = self.worry.df[["Indicator Name", worry_issue]].drop_duplicates()
        cols = ["Country Name", group_country] + list(set(wi["Indicator Name"].values.tolist()) & set(cr.columns.tolist())) 
        cr = cr[cols]
        cr = cr.melt(id_vars=["Country Name", group_country], value_vars=cols[2:], var_name="Indicator Name")
        return cr.merge(wi, on="Indicator Name")
        
        
