import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
st.title("DATA BYTE INDUCTIONS IRIS DATASET jnani kumar 111119132")
st.write("either if you want to predict or add the data to dataset please enter the values below")
st.sidebar.title("you can change the classifier name below")
dataset_name=st.sidebar.title("IRIS DATASET")
classifier_name=st.sidebar.selectbox("select classifier",("KNN","SVM","RANDOM FOREST"))
data=datasets.load_iris()
x=data.data
y=data.target
st.write("shape of iris dataset",x.shape)
st.write("number of classes",len(np.unique(y)))
classes=["setosa","versicolor","virginica"]
def add_parameters(clf_name):
    parameters={}
    if clf_name=="KNN":
        K=st.sidebar.slider("K",1,15)
        parameters["K"]=K
    elif clf_name=="SVM":
        C=st.sidebar.slider("C",0.01,10.0)
        parameters["C"]=C
    else:
        max_depth=st.sidebar.slider("max_depth",2,15)
        n_estimators=st.sidebar.slider("n_estimators",1,100)
        parameters["max_depth"]=max_depth
        parameters["n_estimators"]=n_estimators
    return parameters
params=add_parameters(classifier_name)
def get_classifier(clf_name,params):
    if clf_name=="KNN":
        clf=KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name=="SVM":
        clf=SVC(C=params["C"])
    else:
        clf=RandomForestClassifier(n_estimators=params["n_estimators"],
                                   max_depth=params["max_depth"],random_state=1234)
    return clf
clf=get_classifier(classifier_name,params)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
clf.fit(x_train,y_train)
def prediction(sepal_length,sepal_width,petal_length,petal_width):
    prediction=clf.predict([[sepal_length,sepal_width,petal_length,petal_width]])
    print(prediction)
    st.write("the class is {}".format(classes[prediction[0]]))
    return prediction

#st.title("iris data prediction")
sepal_length=st.text_input("sepal_length",float())
sepal_width=st.text_input("sepal_width",float())
petal_length=st.text_input("petal_length",float())
petal_width=st.text_input("petal_width",float())
z=np.array([[sepal_length,sepal_width,petal_length,petal_width]])
result=""
if st.button("prediction"):
    result=prediction(sepal_length,sepal_width,petal_length,petal_width)
    st.success("the output is {}".format(result))
if st.button("wnated to add_data?? press me"):
    target=st.text_input("target value of above entered data",float())
    np.concatenate([x,z])
    np.concatenate([y,np.array([target])])
    if st.button("add data"):
        st.success("your data is added successfully")
if st.button("about"):
    st.text("iris data prediction for databyte induction")
y_pred=clf.predict(x_test)
acc=accuracy_score(y_test,y_pred)
st.write(f"classifier={classifier_name}")
st.write(f"accuracy={acc}")

#plot
pca=PCA(2)
x_projected=pca.fit_transform(x)
x1=x_projected[:,0]
x2=x_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap="viridis")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.colorbar()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)





    
        
