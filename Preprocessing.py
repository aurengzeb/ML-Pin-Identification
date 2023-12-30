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
df = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/samplerawdata.csv", header=None)
#df = df.fillna(0)
print(df.head())
X = df[df.columns[1:208]]
y = df[df.columns[0:1]]