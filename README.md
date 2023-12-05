# Heart-Disease-Prediction
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing
from google.colab import drive
drive.mount('/content/drive')
import pandas as pd
data = pd.read_csv('/content/Heart_Disease_Prediction.csv')
data
data.shape
data.info()
data.head()
data.tail()
data.describe()
data["Sex"].value_counts()
data["Age"].value_counts()
# See the min, max, mean values
print('The highest hemoglobin was of:',data['Cholesterol'].max())
print('The lowest hemoglobin was of:',data['Cholesterol'].min())
print('The average hemoglobin in the data:',data['Cholesterol'].mean())

import matplotlib.pyplot as plt

# Line plot
plt.plot(data['Cholesterol'])
plt.xlabel("Cholesterol level")
plt.ylabel("Levels")
plt.title("Line Plot")
plt.show()

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(13,5))
data_len=data[data['Heart Disease']==1]['Cholesterol'].value_counts()

ax1.hist(data_len,color='red')
ax1.set_title('Having heart disease')

data_len=data[data['Heart Disease']==0]['Cholesterol'].value_counts()
ax2.hist(data_len,color='green')
ax2.set_title('NOT Having heart disease')

fig.suptitle('Heart Disease')
plt.show()
data.duplicated()
newdata=data.drop_duplicates()
newdata
data.isnull().sum() #checking for total null values
data[1:5]
from sklearn import preprocessing
import pandas as pd

d = preprocessing.normalize(data.iloc[:,1:5], axis=0)
scaled_df = pd.DataFrame(d, columns=["Hemoglobin", "MCH", "MCHC", "MCV"])
scaled_df.head()
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
train,test=train_test_split(data,test_size=0.3,random_state=0,stratify=data['Heart Disease'])
train_X=train[train.columns[:-1]]
train_Y=train[train.columns[-1:]]
test_X=test[test.columns[:-1]]
test_Y=test[test.columns[-1:]]
X=data[data.columns[:-1]]
Y=data['Heart Disease']
len(train_X), len(train_Y), len(test_X), len(test_Y)
model = LogisticRegression()
model.fit(train_X,train_Y)
prediction3=model.predict(test_X)
print('The accuracy of the Logistic Regression is',metrics.accuracy_score(prediction3,test_Y))
report = classification_report(test_Y, prediction3)
print("Classification Report:\n", report)
from sklearn.model_selection import train_test_split #training and testing data split
from sklearn import metrics #accuracy measure
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, classification_report #for confusion matrix
from sklearn.linear_model import LogisticRegression,LinearRegression #logistic regression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
# Create and fit the Linear Regression model
model = LinearRegression()
model.fit(train_X, train_Y)

# Make predictions on the test set
prediction = model.predict(test_X)

# Assuming 'test_Y' contains the true labels for the test set
# Calculate the accuracy
accuracy = accuracy_score(test_Y, prediction.round())

# Print the accuracy
print('The accuracy of Linear Regression is:', accuracy)

#Evaluate the model using various metrices
mse=mean_squared_error(test_Y,prediction)
rmse=mean_squared_error(test_Y,prediction, squared= False) #Calculate square root of mse
mae=mean_absolute_error(test_Y,prediction)
r_squared=r2_score(test_Y,prediction)

print('Mean squared error:',mse)
print('Root mean squared error:',rmse)
print('Mean Absolute Error:',mae)
print('R_squared:',r_squared)
