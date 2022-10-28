#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 10:40:34 2022

@author: vophuoctri
"""

import numpy as np
import pandas as pd

file_name = "gdp(2021).xlsx"
# file_name = 'API_NY.GDP.MKTP.KD_DS2_en_excel_v2_4546386.xls'
sheet_name = "gdp(2021)"
# sheet_name = 'Data'

class GDP():
    def __init__(self, path):
        self.path = path
        self.df = self.get_data('2021')
    def get_data(self, year):
        df = pd.read_excel(self.path+file_name, sheet_name=sheet_name)
        df = df[["Country Name", year]]
        df = df.rename(columns={year: 'gdp'})
        # df['gdp'] = (df[['2019', '2020', '2021']].std(axis=1, skipna=False))/(df[['2019', '2020', '2021']].mean(axis=1, skipna=False))
        return df[['Country Name', 'gdp']]