'''# !/usr/bin/env python
# coding: utf-8
'''
# In[2]:


# get_ipython().system('pip install dataprep')


# # import library

# In[3]:

import json


import pandas as pd
import numpy as np

# data visualization library 
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'


# In[4]:


# dataprep
from dataprep.eda import *
from dataprep.eda.missing import plot_missing
from dataprep.eda import plot_correlation

class Classify():
    #Constructor
    def __init__(self):
        # self.text=text
        self.details={}
        if self is None:
            return 0
        pass

    def routi(self, data):
        # # data analysis

        # In[6]:


        # covid = pd.read_csv('../input/symptoms-and-covid-presence/Covid Dataset.csv')
        covid = pd.read_csv('./Covid Dataset.csv')
        # covid = pd.read_csv('/Covid Dataset.csv')
        covid


        # In[7]:


        covid.info()


        # In[8]:


        covid.describe(include='all')


        # In[9]:


        covid.columns


        # # finding missing value

        # In[10]:


        plot_missing(covid)


        # In[11]:


        # create a table with data missing 
        missing_values=covid.isnull().sum() # missing values

        percent_missing = covid.isnull().sum()/covid.shape[0]*100 # missing value %

        value = {
            'missing_values ':missing_values,
            'percent_missing %':percent_missing  
        }
        frame=pd.DataFrame(value)
        frame


        # like we see our data is clean 0 missing values

        # # data vizualisation

        # ### COVID-19 (target)

        # In[12]:


        sns.countplot(x='COVID-19',data=covid)


        # In[13]:


        covid["COVID-19"].value_counts().plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
        plt.title('number of cases');


        # ### Breathing Problem

        # In[14]:


        sns.countplot(x='Breathing Problem',data=covid)


        # In[15]:


        sns.countplot(x='Breathing Problem',hue='COVID-19',data=covid)


        # ## Fever

        # In[16]:


        sns.countplot(x='Fever',hue='COVID-19',data=covid);


        # ## Dry Cough

        # In[17]:


        sns.countplot(x='Dry Cough',hue='COVID-19',data=covid)


        # ## Sore throat

        # In[18]:


        sns.countplot(x='Sore throat',hue='COVID-19',data=covid)


        # # feature transformation

        # In[19]:


        from sklearn.preprocessing import LabelEncoder
        e=LabelEncoder()


        # In[20]:


        covid['Breathing Problem']=e.fit_transform(covid['Breathing Problem'])
        covid['Fever']=e.fit_transform(covid['Fever'])
        covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
        covid['Sore throat']=e.fit_transform(covid['Sore throat'])
        covid['Running Nose']=e.fit_transform(covid['Running Nose'])
        covid['Asthma']=e.fit_transform(covid['Asthma'])
        covid['Chronic Lung Disease']=e.fit_transform(covid['Chronic Lung Disease'])
        covid['Headache']=e.fit_transform(covid['Headache'])
        covid['Heart Disease']=e.fit_transform(covid['Heart Disease'])
        covid['Diabetes']=e.fit_transform(covid['Diabetes'])
        covid['Hyper Tension']=e.fit_transform(covid['Hyper Tension'])
        covid['Abroad travel']=e.fit_transform(covid['Abroad travel'])
        covid['Contact with COVID Patient']=e.fit_transform(covid['Contact with COVID Patient'])
        covid['Attended Large Gathering']=e.fit_transform(covid['Attended Large Gathering'])
        covid['Visited Public Exposed Places']=e.fit_transform(covid['Visited Public Exposed Places'])
        covid['Family working in Public Exposed Places']=e.fit_transform(covid['Family working in Public Exposed Places'])
        covid['Wearing Masks']=e.fit_transform(covid['Wearing Masks'])
        covid['Sanitization from Market']=e.fit_transform(covid['Sanitization from Market'])
        covid['COVID-19']=e.fit_transform(covid['COVID-19'])
        covid['Dry Cough']=e.fit_transform(covid['Dry Cough'])
        covid['Sore throat']=e.fit_transform(covid['Sore throat'])
        covid['Gastrointestinal ']=e.fit_transform(covid['Gastrointestinal '])
        covid['Fatigue ']=e.fit_transform(covid['Fatigue '])


        # In[21]:


        covid.head()


        # In[22]:


        covid.dtypes.value_counts()


        # 
        # 
        # 
        # 
        # 
        # 

        # # info about our data after transformation 

        # In[23]:


        covid.describe(include='all')


        # In[24]:


        covid.hist(figsize=(20,15));


        # # correlation betwenn features 

        # In[25]:


        plot_correlation(covid)


        # In[26]:


        corr=covid.corr()
        corr.style.background_gradient(cmap='coolwarm',axis=None)


        # # feature selection 

        # #### feature that we gonna delelte :
        # Running Nose / Asthma /Chronic Lung Disease / Headache / Heart Disease / Diabetes / Fatigue / Gastrointestinal / Wearing Masks / Sanitization from Market

        # In[27]:


        covid=covid.drop('Running Nose',axis=1)
        covid=covid.drop('Chronic Lung Disease',axis=1)
        covid=covid.drop('Headache',axis=1)
        covid=covid.drop('Heart Disease',axis=1)
        covid=covid.drop('Diabetes',axis=1)
        covid=covid.drop('Gastrointestinal ',axis=1)
        covid=covid.drop('Wearing Masks',axis=1)
        covid=covid.drop('Sanitization from Market',axis=1)
        covid=covid.drop('Asthma',axis=1)


        # In[28]:


        covid=covid.drop('Fatigue ',axis=1)


        # In[29]:


        corr=covid.corr()
        corr.style.background_gradient(cmap='coolwarm',axis=None)


        # # machine learning algo

        # In[30]:


        from sklearn.model_selection import train_test_split
        from sklearn import metrics
        from sklearn.metrics import accuracy_score


        # In[31]:


        x=covid.drop('COVID-19',axis=1)
        y=covid['COVID-19']
                                    


        # In[32]:


        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20)

        votes={}
        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        custom_test_x=np.array([[int(data['Breathing_Problem']), int(data['Fever']), int(data['Dry_Cough']), int(data['Sore_throat']), int(data['Hyper_Tension']), int(data['Abroad_travel']), int(data['Contact_with_COVID_Patient'])
        , int(data['Attended_Large_Gathering']), int(data['Visited_Public_Exposed_Places']), int(data['Family_working_in_Public_Exposed_Places'])]])#float(e1.get())

        df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])

        algodetails={}


        # ## Logistic Regression

        # In[33]:


        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        #Fit the model
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        #Score/Accuracy
        acc_logreg=model.score(x_test, y_test)*100
        print(acc_logreg)

        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        # df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])
        custom_y_pred = model.predict(df_test_x)
        custom_y_pred
        votes["LR"]=int(custom_y_pred[0])

        # algodetails["LR"]=j{
        #     "acc":int(acc_logreg),
        #     "custom_y_pred":int(custom_y_pred[0])
        # }


        # algodetails["LR"]=json.dumps({
        #     "acc":int(acc_logreg),
        #     "custom_y_pred":int(custom_y_pred[0])
        # })

        algodetails["LR_acc"]=int(acc_logreg)
        algodetails["LR_custom_y_pred"]=int(custom_y_pred[0])
           

        # ##  RandomForestRegressor

        # In[34]:


        #Train the model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=1000)
        #Fit
        model.fit(x_train, y_train)
        #Score/Accuracy
        acc_randomforest=model.score(x_test, y_test)*100
        print(acc_randomforest)

        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        # df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])
        custom_y_pred = model.predict(df_test_x)
        custom_y_pred
        votes["RF"]=int(custom_y_pred[0])

        # algodetails["RF"]={
        #     "acc":int(acc_randomforest),
        #     "custom_y_pred":int(custom_y_pred[0])
        # }


        # algodetails["RF"]=json.dumps({
        #     "acc":int(acc_randomforest),
        #     "custom_y_pred":int(custom_y_pred[0])
        # })

        algodetails["RF_acc"]=int(acc_randomforest)
        algodetails["RF_custom_y_pred"]=int(custom_y_pred[0])



        # ## GradientBoostingRegressor

        # In[35]:


        #Train the model
        from sklearn.ensemble import GradientBoostingRegressor
        GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
        #Fit
        GBR.fit(x_train, y_train)
        acc_gbk=GBR.score(x_test, y_test)*100
        print(acc_gbk)
        
        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        # df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])
        custom_y_pred = GBR.predict(df_test_x)
        custom_y_pred
        votes["GBR"]=int(custom_y_pred[0])

        # algodetails["GBR"]={
        #     "acc":int(acc_gbk),
        #     "custom_y_pred":int(custom_y_pred[0])
        # }


        # algodetails["GBR"]=json.dumps({
        #     "acc":int(acc_gbk),
        #     "custom_y_pred":int(custom_y_pred[0])
        # })

        algodetails["GBR_acc"]=int(acc_gbk)
        algodetails["GBR_custom_y_pred"]=int(custom_y_pred[0])


        # ### KNeighborsClassifier

        # In[36]:


        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors=20)
        knn.fit(x_train, y_train)
        y_pred = knn.predict(x_test)
        #Score/Accuracy
        acc_knn=knn.score(x_test, y_test)*100
        print(acc_knn)

        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        # df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])
        custom_y_pred = knn.predict(df_test_x)
        custom_y_pred
        votes["KNN"]=int(custom_y_pred[0])

        # algodetails["KNN"]={
        #     "acc":int(acc_knn),
        #     "custom_y_pred":int(custom_y_pred[0])
        # }


        # algodetails["KNN"]=json.dumps({
        #     "acc":int(acc_knn),
        #     "custom_y_pred":int(custom_y_pred[0])
        # })

        algodetails["KNN_acc"]=int(acc_knn)
        algodetails["KNN_custom_y_pred"]=int(custom_y_pred[0])


        # ## DecisionTreeClassifier

        # In[37]:


        from sklearn import tree
        t = tree.DecisionTreeClassifier()
        t.fit(x_train,y_train)
        y_pred = t.predict(x_test)
        #Score/Accuracy
        acc_decisiontree=t.score(x_test, y_test)*100
        print(acc_decisiontree)

        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])
        custom_y_pred = t.predict(df_test_x)
        custom_y_pred
        votes["DT"]=int(custom_y_pred[0])

        # algodetails["DT"]={
        #     "acc":int(acc_decisiontree),
        #     "custom_y_pred":int(custom_y_pred[0])
        # }



        # algodetails["DT"]=json.dumps({
        #     "acc":int(acc_decisiontree),
        #     "custom_y_pred":int(custom_y_pred[0])
        # })

        algodetails["DT_acc"]=int(acc_decisiontree)
        algodetails["DT_custom_y_pred"]=int(custom_y_pred[0])


        # x_test

        # y_pred


        # ##  naive_bayes

        # In[38]:


        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(x_train,y_train)
        #Score/Accuracy
        acc_gaussian= model.score(x_test, y_test)*100
        print(acc_gaussian)

        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        # df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])
        custom_y_pred = model.predict(df_test_x)
        custom_y_pred
        votes["NB"]=int(custom_y_pred[0])

        # algodetails["NB"]={
        #     "acc":int(acc_gaussian),
        #     "custom_y_pred":custom_y_pred[0]
        # }


        # algodetails["NB"]=json.dumps({
        #     "acc":int(acc_gaussian),
        #     "custom_y_pred":custom_y_pred[0]
        # })

        algodetails["NB_acc"]=int(acc_gaussian)
        algodetails["NB_custom_y_pred"]=int(custom_y_pred[0])

        # ## svm

        # In[43]:


        #Import svm model
        from sklearn import svm
        #Create a svm Classifier
        clf = svm.SVC(kernel='linear') # Linear Kernel
        #Train the model using the training sets
        clf.fit(x_train, y_train)
        #Predict the response for test dataset
        y_pred = clf.predict(x_test
        )   
        #Score/Accuracy
        acc_svc=clf.score(x_test, y_test)*100
        print(acc_svc)

        


        # y_pred

        # custom_test_x=np.array([[0, 0, 1, 0, 1, 0, 1, 1, 1, 0]])#float(e1.get())
        # df_test_x = pd.DataFrame(data=custom_test_x, index=[0], columns=["Breathing Problem", "Fever", "Dry Cough", "Sore throat", "Hyper Tension", "Abroad travel", "Contact with COVID Patient", "Attended Large Gathering", "Visited Public Exposed Places", "Family working in Public Exposed Places"])
        custom_y_pred = clf.predict(df_test_x)
        custom_y_pred
        votes["SVM"]=int(custom_y_pred[0])

        # algodetails["SVM"]={
        #     "acc":int(acc_svc),
        #     "custom_y_pred":int(custom_y_pred[0])
        # }


        # algodetails["SVM"]=json.dumps({
        #     "acc":int(acc_svc),
        #     "custom_y_pred":int(custom_y_pred[0])
        # })

        algodetails["SVM_acc"]=int(acc_svc)
        algodetails["SVM_custom_y_pred"]=int(custom_y_pred[0])

        


        # print(votes)


        # In[47]:


        print(y_pred[-5:])


        # In[45]:


        print(y_test)


        # In[40]:


        models = pd.DataFrame({
            'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
                    'Random Forest', 'Naive Bayes',   
                    'Decision Tree', 'Gradient Boosting Classifier'],
            'Score': [acc_svc, acc_knn, acc_logreg, 
                    acc_randomforest, acc_gaussian, acc_decisiontree,
                    acc_gbk]})
        models.sort_values(by='Score', ascending=False)

        # return y_pred[-5:]
        # self.details['y_pred']=y_pred
        # return y_pred
        return votes, algodetails


def check(data):# main()
    returnData={}
    # return "ss"
    cl=Classify()

    ## Prediction

    y_pred, details=cl.routi(data)

    #print(y_pred)
    # print(details)
    # cl.routi()
    # y_pred=cl.details
    # print("pred_value:")
    # print(y_pred)


    # lists = y_pred.tolist()
    # json_str = json.dumps(lists)
    json_str = json.dumps(y_pred)
    returnData['y_pred-votes']= json_str

    # lists = details.tolist()
    # json_str = json.dumps(lists)
    # details=list(set(details))
    json_str = json.dumps(details)
    returnData['y_pred-algo_details']= json_str #details


    # returnData['y_pred']= y_pred#json_str


    # returnData['y_pred-votes']=y_pred
    # returnData['y_pred-algo_details']=details
    
    ##


    return returnData
    # return routi()

