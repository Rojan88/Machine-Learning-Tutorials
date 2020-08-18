import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

cancer=datasets.load_breast_cancer()
X=cancer.data
y=cancer.target
X_train,X_test,y_train,y_test=sklearn.model_selection.train_test_split(X,y,test_size=0.2)
#classes=['malignant','benign']

clf=svm.SVC()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
acc=metrics.accuracy_score(y_test,y_pred)
print(acc)
