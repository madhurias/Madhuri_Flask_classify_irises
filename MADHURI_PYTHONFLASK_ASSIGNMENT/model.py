#importing libraries
import numpy as np
import pandas as pd
from pandas import ExcelFile 
import xlrd 
import pickle
from sklearn.model_selection import train_test_split
iris_data=pd.read_excel("iris.xls")
from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
iris_data['Classification']=label_encoder.fit_transform(iris_data['Classification'])
y=iris_data['Classification']
x=iris_data.drop(['Classification','PW'],axis=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.linear_model import LogisticRegression
log_model=LogisticRegression()
log_model=log_model.fit(x_train,y_train)
y_pred=log_model.predict(x_test)
#from sklearn.svm import SVC
#svm_cls=SVC(kernel='linear')
#svm_cls.fit(x_train,y_train)
#y_pred_svm=svm_cls.predict(x_test)
accuracy_score(y_test,y_pred)
confusion_matrix(y_test,y_pred)
pickle.dump(log_model,open('model.pkl','wb'))


