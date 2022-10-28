#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 20:01:46 2022

@author: vophuoctri
"""

import numpy as np
import pandas as pd

file_name = "gdp(2021).xlsx"
sheet_gdp = "gdp(2021)"
sheet_country = "Metadata - Countries"

class Country():
    def __init__(self, path):
        self.path = path
        self.df = self.get_data()
        self.income_group = self.df["IncomeGroup"].unique().tolist()
        self.region = self.df["Region"].unique().tolist()
    def get_data(self):
        country1 = pd.read_excel(self.path+file_name, sheet_name=sheet_gdp)[["Country Name", "Country Code"]]
        country2 = pd.read_excel(self.path+file_name, sheet_name=sheet_country)[["Country Code", "Region", "IncomeGroup"]]
        self.country = country1.merge(country2, on="Country Code")
        for i in range(len(self.country)):
            txt = self.country.iloc[i]
            if txt["Country Name"] in self.country["IncomeGroup"].unique():
                txt["IncomeGroup"] = txt["Country Name"]
            if txt["Country Name"] in self.country["Region"].unique():
                txt["Region"] = txt["Country Name"]
        return self.country