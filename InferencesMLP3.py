# Imports
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
from keras.layers.core import Dense, Activation, Dropout
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from statistics import mean
from sklearn.metrics import precision_recall_fscore_support
import logging
import pandas as pd
from keras.models import load_model

model = load_model('C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/TCBDL.h5')
#model = load_model('TCBDL.h5')
#df = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/new_dataxhead.csv", header=None)
df = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/new_data.csv")
is_U0 = df['complevel_1']=="U0"
U0only = df[is_U0]
Filteredcolumn = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/FilteredColumn.csv")
Filter = list(Filteredcolumn.columns.values.tolist())
Preprocessing = U0only[Filter]
theshape = Preprocessing.shape
Preprocessing = Preprocessing.fillna(0)
X = Preprocessing[Preprocessing.columns[1:theshape[1]]]
y = Preprocessing[Preprocessing.columns[0:1]]
y = y.values.tolist()
mm = X.values.tolist()
mmax = max(mm[0])
y_new = []
for i in y:
    y_new.append(i[0])

all_X = np.array(X, dtype=np.float)
all_X = all_X/mmax
all_y = np.array(y_new, dtype=np.int)

# Convert target classes to categorical ones
#Y_test = to_categorical(all_y, num_classes)

y_inferencse = model.predict(all_X)
xx = y_inferencse.shape
Infered = []
for i in range (xx[0]):
    if y_inferencse[i][0]>y_inferencse[i][1]:
        Infered.append(0)
    else:
        Infered.append(1)

#data = [{'a': all_y, 'b': Infered}]
df = pd.DataFrame({
    'Real': all_y,
    'Predicted': Infered
})
df.to_csv('ResultInferences.csv', index=False, na_rep='Unknown')

acc = accuracy_score(all_y, Infered)
p2a,p2b,p2c,p2d = precision_recall_fscore_support(all_y, Infered,pos_label = 0, average='binary')
print(acc,p2a,p2b,p2c)
