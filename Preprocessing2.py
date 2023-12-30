#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      asmohdsa
#
# Created:     28/04/2021
# Copyright:   (c) asmohdsa 2021
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd
df = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/raw_data.csv")
is_U0 = df['complevel_1']=="U0"
U0only = df[is_U0]
Filteredcolumn = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/FilteredColumn.csv")
Filter = list(Filteredcolumn.columns.values.tolist())
Preprocessing = U0only[Filter]
theshape = Preprocessing.shape
Preprocessing = Preprocessing.fillna(0)
X = Preprocessing[Preprocessing.columns[1:theshape[1]]]
y = Preprocessing[Preprocessing.columns[0:1]]



