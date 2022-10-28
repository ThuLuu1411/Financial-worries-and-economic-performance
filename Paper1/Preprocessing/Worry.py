#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:11:04 2022

@author: thuluu
"""

import numpy as np
import pandas as pd
import re

file_name = "Databank-wide.xlsx"
sheet_name = "Series_Table"

class Worry():
    def __init__(self, path):
        self.path = path
        self.df = self.get_data()
        self.indicator_name = self.df["Indicator Name"].unique().tolist()
        self.maingroup = self.df["maingroup"].unique().tolist()
        self.subgroup = self.df["subgroup"].unique().tolist()
        self.how = self.df["how"].unique().tolist()
        self.branch = self.df["branch"].unique().tolist()
    def indicators_get(self):
        self.indicators = pd.read_excel(self.path+file_name, sheet_name=sheet_name)[["Series", "Indicator Name"]]
        return self.indicators
    def process1_get(self):
        self.process1 = self.indicators_get()
        for i in range(self.process1.shape[0]):
            if "," not in self.process1.loc[i]["Indicator Name"]:
                self.process1.loc[i]["Indicator Name"] = self.process1.loc[i]["Indicator Name"].replace(" (% age 15+)", ", all (% age 15+)")
        return self.process1
    def process2_get(self):
        self.process2 = self.process1_get()
        for i in range(self.process2.shape[0]):
            txt = self.process2.loc[i]["Series"]
            x = re.search("\d$", txt)
            if x:
                continue
            else:
                self.process2.loc[i]["Series"] = txt + ".0"
        return self.process2
    def process3_get(self):
        process3 = self.process2_get()
        worry_ls = []

        for i in process3["Series"]:
            if "fin44" in i or "fin45" in i:
                worry_ls.append(i)

        self.worry_df = process3[process3["Series"].isin(worry_ls)].reset_index().drop("index", axis=1)
        
        self.worry_df["maingroup"] = np.nan
        self.worry_df["subgroup"] = np.nan
        self.worry_df["how"] = np.nan
        self.worry_df["branch"] = np.nan


        for i in range(self.worry_df.shape[0]):
            txt = self.worry_df.loc[i]["Indicator Name"]
            txt = txt.replace(":", ",")
            worried = re.search("about", txt)
            covid19 = re.search("COVID-19", txt)
            most = re.search("Most worrying financial issue", txt)
            
            if worried:
                txt = txt.replace("about", ",")
                self.worry_df["maingroup"].loc[i] = txt.split(",")[0]
                self.worry_df["subgroup"].loc[i] = txt.split(",")[1]
                self.worry_df["how"].loc[i] = txt.split(",")[2]
                self.worry_df["branch"].loc[i] = txt.split(",")[3]

                
            if covid19:
                self.worry_df["maingroup"].loc[i] = txt.split(",")[0]
                self.worry_df["how"].loc[i] = txt.split(",")[1]
                self.worry_df["branch"].loc[i] = txt.split(",")[2]
                
            if most:
                self.worry_df["maingroup"].loc[i] = txt.split(",")[0]
                self.worry_df["subgroup"].loc[i] = txt.split(",")[1]
                self.worry_df["branch"].loc[i] = txt.split(",")[2]
        self.worry_df["how"] = self.worry_df["how"].replace(" not worried at all", ' not worried')
        return self.worry_df
               
    def get_data(self):
        self.df = self.process3_get()
        for i in range(self.df.shape[0]):
            if "," in self.df.loc[i]["Indicator Name"]:
                self.df.loc[i]["Indicator Name"] = self.df.loc[i]["Indicator Name"].replace(", all (% age 15+)", " (% age 15+)")

        return self.df
        

