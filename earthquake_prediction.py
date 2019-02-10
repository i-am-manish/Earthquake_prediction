import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set()

data = pd.read_csv("../input/train.csv",dtype={"acoustic_data": np.float32, "time_to_failure": np.float32}) #data is of 8.6GB, so nrows needs to be set
data.head(10)
#acoustic_data=signal & time_to_failure=quaketime


#EDA # i'll take these features to train
data.describe()

data.shape

#make learnable features (empty yet)
nrows=200000
sets = int(np.floor(data.shape[0] / nrows))

X_train = pd.DataFrame(index=range(sets),columns = ['mean','std','min-0quat','25quat','50quat','75quat','max-1quat'])
y_train = pd.DataFrame(index=range(sets),columns = ['time_to_failure'])



#filling the tables
for set in (range(sets)):   #tqdm(range(sets))
    x=data.iloc[set*nrows:set*nrows+nrows]       #s.iloc[:3] returns us the first 3 rows (since it treats 3 as a position) and s.loc[:3] returns us the first 8 rows (since it treats 3 as a label)
    y = x['time_to_failure'].values[-1]
    x = x['acoustic_data'].values
    #adding mean,max,std to their respective columns(features)
    X_train.loc[set,'mean']=np.mean(x)
    X_train.loc[set,'std']=np.std(x)
    X_train.loc[set,'min-0quat']=np.quantile(x,0.01)
    X_train.loc[set,'25quat']=np.quantile(x,0.25)
    X_train.loc[set,'50quat']=np.quantile(x,0.5)
    X_train.loc[set,'75quat']=np.quantile(x,0.75)
    X_train.loc[set,'max-1quat']=np.quantile(x,0.99)
    y_train.loc[set,'time_to_failure'] = y



X_train.head(5)

y_train.head(5)


from sklearn.preprocessing import StandardScaler
#it will transform your data such that its distribution will have a mean value 0 and standard deviation of 1.
#Given the distribution of the data, each value in the dataset will have the sample mean value subtracted,
#and then divided by the standard deviation of the whole dataset.
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X_train)




from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


model=Sequential()
model.add(Dense(32,input_shape=(7,),activation = "relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(32,activation="relu"))
model.add(Dense(1))
adam=Adam(lr=0.0001)
model.compile(loss="mae",optimizer=adam)

model.fit(X_scaled,y_train,epochs = 500,validation_split=0.2)#batch_size




submission_data = pd.read_csv('../input/sample_submission.csv',index_col = 'seg_id')
submission_data.tail(5)



X_test = pd.DataFrame(columns = X_train.columns,dtype = np.float32,index=submission_data.index)
X_test.head(4)

from tqdm import tqdm
for seq in tqdm(X_test.index):
    test_data = pd.read_csv('../input/test/'+seq+'.csv')
    x = test_data['acoustic_data'].values
    X_test.loc[seq,'mean'] = np.mean(x)
    X_test.loc[seq,'std']  = np.std(x)
    X_test.loc[set,'min-0quat']=np.quantile(x,0.01)
    X_test.loc[set,'25quat']=np.quantile(x,0.25)
    X_test.loc[set,'50quat']=np.quantile(x,0.5)
    X_test.loc[set,'75quat']=np.quantile(x,0.75)
    X_test.loc[set,'max-1quat']=np.quantile(x,0.99)



X_test_scaled = scaler.transform(X_test)
#prediction
pred=model.predict(X_test_scaled)

#KAGGLE SUBMISSION
print(submission_data.shape)
submission_data.head()

print(pred.shape)
pred

submission_data['seg_id'] = submission_data.index

pred=submission_data["time_to_failure"]

submission_data.to_csv('sub_earthquake.csv',index = False)

