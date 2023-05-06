#!/usr/bin/env python
# coding: utf-8

# # Predicting whether a patient has a low or high chance of having a heart attack according to her/his medical information

# ### All four classifiers(KNN, decision tree,svm,logistic regression) are used in this project to determine the best model.
# ### svm(linear kerneling) and logistic regression gives the highest accuracy than the other classifires and it seems this dataset is separable by a sigmoid function(LR) or a hyperplane(svm) in higher dimensions.
# 

# In[1]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv(r"C:\Users\mohad\Downloads\16325569497737558.csv")
df.head()
#print(df['output'].value_counts())


# In[3]:


x=df[['age','sex','cp','trtbps','chol','fbs','restecg','thalachh','exng','oldpeak','slp','thall','caa']].values
x[:5]


# In[4]:


y=df[['output']].values
y[:5]


# In[5]:


df.shape
df.dtypes
print(df['caa'].value_counts()) 


# ### Removing value 4 of caa column

# In[100]:


indexes=df.index[df['caa'] == 4].tolist()
print(indexes)
df=df.drop(df.index[indexes])   
print(df['caa'].value_counts())


# In[7]:


ax =df[df['output'] == 0][0:50].plot(kind='scatter', x='age', y='trtbps', color='DarkBlue', label='low chance');
df[df['output'] == 1][0:50].plot(kind='scatter', x='age', y='trtbps', color='Yellow', label='high chance', ax=ax);
plt.show()


# ## SVM Model

# ### creating a loop through different kernel functions to see which one has the higher accuracy.
# ### the linear kerneling gives an f1 score as 0.92 which is the best for this dataset.

# ### scaling and fitting data on normal distribution chart. because some features like age and chol differ greatly in their range and in svm modeling we need to calculate the distance of data points.

# In[6]:


from sklearn import preprocessing
scaler=preprocessing.StandardScaler()
x = scaler.fit(x).transform(x)
x[0:5]


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
print ('Train set:', x_train.shape,  y_train.shape)
print ('Test set:', x_test.shape,  y_test.shape)


# In[9]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# ### training model and evaluation by test dataset.

# In[12]:


from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix
import itertools
kernels=['rbf','linear','sigmoid','poly']
for kernelf in kernels:
    svmModel=svm.SVC(kernel=kernelf)
    svmModel.fit(x_train,y_train)
    yhat=svmModel.predict(x_test)
    print(yhat[:5])
    print(y_test[:5])
    cnf_matrix = confusion_matrix(y_test, yhat, labels=[0,1])
    np.set_printoptions(precision=2)
    print("kernel function:"+kernelf)
    print (classification_report(y_test, yhat))
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['high chance=1','low chance=0'],title='Confusion matrix')
    


# ## Decision Tree Model

# In[17]:


from sklearn.tree import DecisionTreeClassifier
treeModel=DecisionTreeClassifier(criterion='entropy',min_samples_split=6)
treeModel.fit(x_train,y_train)
yhat2=treeModel.predict(x_test)
print(yhat[:5])
print(y_test[:5])


# ## Evaluation

# In[18]:


from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, yhat))
print (classification_report(y_test, yhat2))
cnf_matrix = confusion_matrix(y_test, yhat2, labels=[0,1])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['high chance=1','low chance=0'],title='Confusion matrix')


# ## KNN Model

# ## Evaluation
# ### creating a loop through different k values to determine the value which gives the highest accuracy.

# In[21]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
ks=11
k_values=np.zeros((ks-1))
acc_score=np.zeros((ks-1))
for k in range(1,ks):
    knnModel=KNeighborsClassifier(n_neighbors = k).fit(x_train,y_train)
    yhat3=knnModel.predict(x_test)
    acc_score[k-1]=metrics.accuracy_score(y_test, yhat3)
    k_values[k-1]=k
    print("Accuracy score with k="+str(k)+" is "+str(acc_score[k-1]))
    print (classification_report(y_test, yhat3))
    cnf_matrix = confusion_matrix(y_test, yhat3, labels=[0,1])
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=['high chance=1','low chance=0'],title='Confusion matrix')
print("the best accuracy score is "+ str(acc_score.max())+" with k : "+str(k_values.max()))


# ## LogisticRegression Model

# In[34]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LRModel = LogisticRegression(C=0.01, solver='newton-cholesky').fit(x_train,y_train)


# In[35]:


yhat4 = LRModel.predict(x_test)
print(yhat4[:5])
print(y_test[:5])


# ## Evaluation

# In[36]:


from sklearn.metrics import classification_report, confusion_matrix
print (classification_report(y_test, yhat4))
cnf_matrix = confusion_matrix(y_test, yhat4, labels=[0,1])
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['high chance=1','low chance=0'],title='Confusion matrix')


# In[ ]:




