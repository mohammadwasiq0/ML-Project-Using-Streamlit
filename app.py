import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# About App
st.write('''
# Explore Different Machine Learning Models with Datasets by Mohammad Wasiq
Let's see which model is best
''')

# Datasets
dataset_name= st.sidebar.selectbox(
    'Select Dataset',
    ('Iris', 'Brest Cancer', 'Wine')
)

# Classifier Model
classifier_name= st.sidebar.selectbox(
    'Select Classifier',
    ('KNN', 'SVM', 'Random Forest')
)

# Function for dataset
def get_dataset(dataset_name):
    data = None
    if dataset_name == "Iris":
        data= datasets.load_iris()
    elif dataset_name== "Wine":
        data= datasets.load_wine()
    else:
        data= datasets.load_breast_cancer()

    x= data.data
    y= data.target

    return x, y

X, y= get_dataset(dataset_name)

# Shape of dataset
st.write("Shape of Dataset : ", X.shape)
st.write("Number of Classes : ", len(np.unique(y)))

# Classifier parameters
def add_parameters_ui(classifier_name):
    params= dict() # Create an empty dictionary
    if classifier_name== 'SVM':
        C= st.sidebar.slider('C', 0.01, 10.0)
        params['C']= C # It's the degree of correct classifier
    elif classifier_name== 'KNN':
        K= st.sidebar.slider('K', 1, 15)
        params['K']= K # It is the number of nearest neighbour
    else:
        max_depth= st.sidebar.slider('max_depth', 2, 15)
        params['max_depth']= max_depth # depth of every tree that grow in Random Forest
        n_estimators= st.sidebar.slider('n_estimators', 1, 100)
        params['n_estimators']= n_estimators # Number of trees

    return params

# Call the function
params= add_parameters_ui(classifier_name)

# Classifiers
def get_classifier(classifier_name, params):
    clf= None
    if classifier_name == 'SVM':
        clf= SVC(C= params['C'])
    elif classifier_name == 'KNN':
        clf= KNeighborsClassifier(n_neighbors= params['K'])
    else:
        clf= RandomForestClassifier(n_estimators= params['n_estimators'],
                                    max_depth=params['max_depth'], random_state=1234)

    return clf

clf= get_classifier(classifier_name, params)

# Train Test Split of data
X_train, X_text, y_train, y_test= train_test_split(X, y, train_size= 0.2, random_state= 1234)

clf.fit(X_train, y_train)

y_pred= clf.predict(X_text)

# Model Accuracy
acc= accuracy_score(y_test, y_pred)
st.write(f'Classifier= {classifier_name}')
st.write(f'Accuracy = ', acc)

# Plotting the Data
pca= PCA(2)
X_projected= pca.fit_transform(X)

# Dimension
x1= X_projected[:, 0]
x2= X_projected[:, 1]

fig= plt.figure()
plt.scatter(x1, x2,
            c= y, alpha= 0.8,
            cmap= 'viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()

# plt.show(
st.pyplot(fig)
