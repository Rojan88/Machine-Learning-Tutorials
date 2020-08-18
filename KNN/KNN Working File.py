import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model,preprocessing

data=pd.read_csv('./car.data')
print(data.head())

le=preprocessing.LabelEncoder()

buying=le.fit_transform(list(data['buying']))
maint=le.fit_transform(list(data['maint']))
doors=le.fit_transform(list(data['doors']))
persons=le.fit_transform(list(data['persons']))
lug_boot=le.fit_transform(list(data['lug_boot']))
safety=le.fit_transform(list(data['safety']))
cls=le.fit_transform(list(data['class']))

predict='class'

X=list(zip(buying,maint,doors,persons,lug_boot,safety))
y=list(cls)

X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2)

model=KNeighborsClassifier(n_neighbors=5)
model.fit(X_train,y_train)
acc=model.score(X_test,y_test)
print(acc)
predicted=model.predict(X_test)
names=['unacc','acc','good','vgood']
for x in range(len(predicted)):
	print('Predicted:',names[predicted[x]],'Data:',X_test[x],'Actual:',names[y_test[x]])

n=model.kneighbors([X_test[x]],9,True)
print('N:',n)