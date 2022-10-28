#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 18:30:02 2022

@author: vophuoctri
"""

import numpy as np
import pandas as pd
from Preprocessing.Country import Country
from Preprocessing.Worry import Worry
from Preprocessing.GDP import GDP
from data import Finance
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


path = "/Users/vophuoctri/OneDrive/quaypha/data/"
ls_worry = ["Indicator Name", "maingroup", "subgroup", "branch", "how"]

class Visualization():
    def __init__(self, path, group_country, worry_columns):
        self.path = path
        self.data = Finance(path)
        self.df = self.get_data(group_country, worry_columns)
    def get_data(self, group_country, worry_columns):
        df = self.data.country_worry(group_country, worry_columns)
        ls = df[group_country].unique()
        df = df[df["Country Name"].isin(ls)]
        df["subgroup"] = df["subgroup"].fillna("No value")
        return df
    def data_maingroup(self, maingroup):
        return self.df[self.df["maingroup"]==maingroup]
    def data_subgroup(self, maingroup, subgroup):
        df = self.data_maingroup(maingroup)
        return df[df["subgroup"]==subgroup]
    def data_branch_duel_maingroup(self, maingroup, branch_duel):
        df = self.data_maingroup(maingroup)
        return df[df["branch"].isin(branch_duel)]
    def data_branch_duel_subgroup(self, maingroup, subgroup, branch_duel):
        df = self.data_subgroup(maingroup, subgroup)
        return df[df["branch"].isin(branch_duel)]
    def change_data_maingroup(self, maingroup, branch_duel):
        df = self.data_branch_duel_maingroup(maingroup, branch_duel)
        ls_subgroup = df.subgroup.unique().tolist()
        dfdf = df.pivot_table(values="value", index=["Country Name", "branch"],
                              columns = "subgroup", dropna=False).reset_index()
        for i in range(len(ls_subgroup)):
            try:
                dfdf[ls_subgroup[i]] = dfdf[ls_subgroup[i]] + dfdf[ls_subgroup[i-1]]
            except:
                continue
        return dfdf, ls_subgroup
    def change_data_subgroup(self, maingroup, subgroup, branch_duel):
        df = self.data_branch_duel_subgroup(maingroup, subgroup, branch_duel)
        dfdf = df.pivot_table(values="value", index=["Country Name", "branch"], 
                               columns="how", dropna=False).reset_index().fillna(0)
        dfdf[" very worried"] = dfdf[" very worried"] + dfdf[" not worried"] + dfdf[" somewhat worried"]
        dfdf[" somewhat worried"] = dfdf[" not worried"] + dfdf[" somewhat worried"]
        return dfdf
    def barplot_maingroup(self, maingroup, branch_duel):
        df = self.change_data_maingroup(maingroup, branch_duel)
        ls_subgroup= df[1][::-1]
        df = df[0]
        plt.figure(figsize=(10,5))
        df["no value"]=1
        ls_color = sns.color_palette()
        handles = []
        for j,i in enumerate(ls_subgroup):
            sns.barplot(x = "Country Name", y=i,
                        data = df[["Country Name", "branch", i]],
                        hue="branch", palette = [ls_color[j]],
                        edgecolor="white")
            handles.append(mpatches.Patch(color=ls_color[j], label=i))
        for i,j in enumerate(df["branch"].unique(),1):
            col = "Col-"+ str(i) + ": " + str(j)
            handles.append( mpatches.Patch(color="white", label=col))
        plt.ylabel("worried issue")
        plt.legend(#title = columns_bar,
                   handles=handles, 
                   bbox_to_anchor=(1.05, -0.1), ncol=2)
        plt.title(maingroup)
        return plt.show()
        
    def barplot_subgroup(self, maingroup, subgroup, branch_duel):
        df = self.change_data_subgroup(maingroup, subgroup, branch_duel)
        plt.figure(figsize=(10,5))
        df["no value"] = 1
        ls_color = sns.color_palette()
        s4 = sns.barplot(x = 'Country Name', y = 'no value', 
                         data = df[["Country Name", "branch", "no value"]],
                         hue="branch", palette = ["grey"],
                         edgecolor="white")
        grey_patch = mpatches.Patch(color="grey", label='No Value')
        s1 = sns.barplot(x = 'Country Name', y = ' very worried', 
                         data = df[["Country Name", "branch", " very worried"]],
                         hue="branch", palette = [ls_color[1]],
                         edgecolor="white")
        blue_patch = mpatches.Patch(color=ls_color[1], label='Very Worried')
        s2 = sns.barplot(x = 'Country Name', y = ' somewhat worried', 
                         data = df[["Country Name", "branch", " somewhat worried"]],
                         hue="branch", palette = [ls_color[2]],
                         edgecolor="white")
        red_patch = mpatches.Patch(color=ls_color[2], label='Somewhat Worried')
        s3 = sns.barplot(x = 'Country Name', y = ' not worried', 
                         data = df[["Country Name", "branch", " not worried"]],
                         hue="branch", palette = [ls_color[3]],
                         edgecolor="white")
        green_patch = mpatches.Patch(color=ls_color[3], label='Not Worried')
        plt.ylabel("how")
        plt.xlabel("")
        columns_bar = []
        handles=[grey_patch, blue_patch, red_patch, green_patch]
        for i,j in enumerate(df["branch"].unique(),1):
            col = "Col-"+ str(i) + ": " + str(j)
            handles.append( mpatches.Patch(color="white", label=col))
        plt.legend(#title = columns_bar,
                   handles=handles, 
                   bbox_to_anchor=(0.75, -0.1), ncol=2)
        if subgroup != "No value":
            plt.title(maingroup + "" + subgroup)
        else:
            plt.title(maingroup + "" )
        return plt.show()
    def barplot(self, maingroup, branch_duel):
        df = self.data_maingroup(maingroup)
        if all(df["how"].isnull()):
            return self.barplot_maingroup(maingroup, branch_duel)
        else:
            for i in df["subgroup"].unique():             
                self.barplot_subgroup(maingroup, i, branch_duel)

visual = Visualization(path, "IncomeGroup", ls_worry)
maingroup = visual.data.worry.maingroup
subgroup = visual.data.worry.subgroup + ["No value"]
how = visual.data.worry.how
branch = visual.data.worry.branch

branch_duel = [ ' in labor force (% age 15+)',   ' out of labor force (% age 15+)']

for i in maingroup:
    visual.barplot(i, branch_duel)
            