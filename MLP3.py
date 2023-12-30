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

num_classes = 2

#df = pd.read_csv("C:/Users/asmohdsa/OneDrive - Intel Corporation/Documents/IMR/TRB ML/Dataprocessingxhead.csv", header=None)
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
Y_train = to_categorical(all_y, num_classes)
#Y_test = to_categorical(Y_test, num_classes)
feature_vector_length = all_X.shape[1]
# Set the input shape
input_shape = (feature_vector_length,)
model = Sequential()
model.add(Dense(2*all_X.shape[1]+1, input_shape=input_shape, activation='relu'))
model.add(Dense(int((2*all_X.shape[1]+1)/1), activation='relu'))
model.add(Dense(int((2*all_X.shape[1]+1)/1), activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#Accuracy,Precision,Recall,F1 Initialization
acct = []
pM = []
pm = []
pW = []
rM = []
rm = []
rW = []
FM = []
Fm = []
FW = []

kf = KFold(n_splits=10)
kf.get_n_splits(all_X)
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, filename='AccuracyPrecisionRecallFscoreBinary.log')
for train_index,test_index in kf.split(all_X):
#for train_index, test_index in skf.split(all_X, Y_train):
    X_train, X_test = all_X[train_index], all_X[test_index]
    y_train, y_test = Y_train[train_index], Y_train[test_index]
    # Create the model
    model.fit(X_train, y_train, epochs=10, batch_size=250, verbose=1, validation_split=0.1)
    Infered = []
    testreshape = []
    y_inferencse = model.predict(X_test)
    xx = y_inferencse.shape
    for i in range (xx[0]):
        if y_inferencse[i][0]>y_inferencse[i][1]:
            Infered.append(0)
        else:
            Infered.append(1)


    for i in range (xx[0]):
        if y_test[i][0]>y_test[i][1]:
            testreshape.append(0)
        else:
            testreshape.append(1)

    acc = accuracy_score(testreshape, Infered)
    acct.append(acc)
    p1a,p1b,p1c,p1d = precision_recall_fscore_support(testreshape, Infered, average='micro')
    p2a,p2b,p2c,p2d = precision_recall_fscore_support(testreshape, Infered,pos_label = 0, average='binary')
    p3a,p3b,p3c,p3d = precision_recall_fscore_support(testreshape, Infered, average='weighted')
    logging.info("FoldingAccuracy:%f,PrecisionMicro:%f,RecallMicro:%f,FBetaMicro:%f,PrecisionMacro:%f,RecallMacro:%f,FBetaMacro:%f,PrecisionW:%f,RecallW:%f,FBetaW:%f" %(acc,p1a,p1b,p1c,p2a,p2b,p2c,p3a,p3b,p3c))
    print("FoldingAccuracy:%f,PrecisionMicro:%f,RecallMicro:%f,FBetaMicro:%f,PrecisionMacro:%f,RecallMacro:%f,FBetaMacro:%f,PrecisionW:%f,RecallW:%f,FBetaW:%f" %(acc,p1a,p1b,p1c,p2a,p2b,p2c,p3a,p3b,p3c))
    pm.append(p1a)
    rm.append(p1b)
    Fm.append(p1c)
    pM.append(p2a)
    rM.append(p2b)
    FM.append(p2c)
    pW.append(p3a)
    rW.append(p3b)
    FW.append(p3c)
model.save('TCBDL.h5')  # creates a HDF5 file 'my_model.h5'
AccuracyTesting = mean(acct)
pmt =  mean(pm)
rmt =  mean(rm)
Fmt =  mean(Fm)
pMtt =  mean(pM)
rMtt =  mean(rM)
FMtt =  mean(FM)
pWt =  mean(pW)
rWt =  mean(rW)
FWt =  mean(FW)
logging.info("AFoldingAccuracy:%f,APrecisionMicro:%f,ARecallMicro:%f,AFBetaMicro:%f,APrecisionMacro:%f,ARecallMacro:%f,AFBetaMacro:%f,APrecisionW:%f,ARecallW:%f,AFBetaW:%f" %(AccuracyTesting,pmt,rmt,Fmt,pMtt,rMtt,FMtt,pWt,rWt,FWt))




