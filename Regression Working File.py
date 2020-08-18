import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data=pd.read_csv('student-mat.csv',sep=';')
data=data[['G1','G2','G3','studytime','failures','absences']]

X=np.array(data.drop(['G3'],1))
y=np.array(data['G3'])
X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0..2)

linear=linear_model.LinearRegression()
linear.fit(X_train,y_train)
acc=linear.score(X_test,y_test)
print(acc)
print('Coefficient:\n',linear.coef_)
print('\nIntercept:\n',linear.intercept_)
predictions=linear.predict(X_test)
for x in range(len(predictions)):
	print(predictions[x],X_test[x],y_test[x])

with open('studentmodel.pickle','wb') as f:
	pickle.dump(linear,f)
pickle_in=open('studentmodel.pickle','rb')
linear=pickle.load(pickle_in)

style.use('ggplot')
pyplot.scatter(data['G2'],data['G3'])
pyplot.xlabel('rojan')
pyplot.ylabel('upreti')
pyplot.show()
