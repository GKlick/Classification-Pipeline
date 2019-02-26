
# coding: utf-8
# coding: utf-8

# In[13]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

import plotly.graph_objs as go
#use this format for working locally 
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot, plot_mpl
init_notebook_mode(connected=True)
from ipywidgets import interactive, FloatSlider


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import naive_bayes
from sklearn.metrics import accuracy_score, classification_report
import imblearn.over_sampling
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import cross_validate
from sklearn.metrics.scorer import make_scorer


# In[1]:


def grid_search_return_scores(X_data, y_data, model_, param_grid, kfolds = 5):
    '''
    Given data, model, parameters, number of kfolds, and scoring metric, 
    returns best parameters for fitted model

    Parameters:
        X_data: X data to fit on
        Y_data: Y data to fit on
        Model_: Choice of model
        Param_grid: parameters, as a dictionary
        kfolds: Number of kfolds within cross validation

    Returns:
        Optimal parameters for model
    '''
    grid = GridSearchCV(model_,param_grid, cv=kfolds, scoring = 'roc_auc')
    grid.fit(X_data, y_data)

    
    return grid.best_params_


# In[14]:


def cross_val(model,X_data, y_data):
    '''
    Performs cross validation on model using ROC and F1 as scoring metrics. Prompts user for number of kfolds

    Parameters:
        Model: Model with parameters set
        X_data: X data
        Y_data: y data

    Returns:
        Prints ROC and F1 score of cross validation
    '''
    
    kfolds = int(input('How many folds for cross validation? '))
    
    roc = cross_val_score(model, X_data, y_data, cv = kfolds, scoring = 'roc_auc').mean()
    f1 = cross_val_score(model, X_data, y_data, cv = kfolds, scoring = 'f1').mean()
    print('ROC AUC Score for cross validation: %s'%(roc))
    print('F1 Score for cross validation: %s\n'%(f1))


# In[15]:


def make_confusion_matrix(model_,X_data, y_data):
    '''
    Creates a confusion matrix based off the model predictions
    
    Parameters:
    Model: Model with parameters set
    X_data: X data
    Y_data: y data
    
    Returns:
        Prints confusion matrix and precision score
    '''
    y_predict = model_.predict(X_data)
    confusion_matrix(y_data, y_predict)
    plt.figure(figsize = (10,7))
    print('Confusion Matrix')
    sns.heatmap(confusion_matrix(y_data, y_predict),annot=True, cmap = 'Blues', fmt='g',
           xticklabels=['Actual', 'Actual'],
           yticklabels=['Predicted', 'Predicted']);
    print('Precision Score %s'%(precision_score(y_data, y_predict, average='binary')))


# In[16]:


def model_score(model_, X_data, y_data):
    '''
    Model_score is a one stop shop for all your modeling needs.
    Feed in a model with parameters set, as well as the data to fit on.

    Watch in amazement as you receive a fitted model, error metrics, cross validation(user will be prompted for this input), and confusion matrix

    Parameters:
        Model: Model with parameters set
        X_data: X data
        Y_data: y data

    Returns:
        Prints error metrics and confusion matrix, returns fitted model

    '''

    model_fit = model_.fit(X_data, y_data)
    y_predict = model_fit.predict(X_data)
    
    print('ROC Score for the model: %s'%(roc_auc_score(y_data, y_predict)))
    print('F1 Score for the model: %s\n'%(f1_score(y_data, y_predict)))
    
    #This calls two other functions, that print out there values
    cross_val(model_fit, X_data, y_data)
    make_confusion_matrix(model_fit,X_data, y_data)
    
    return model_fit


# In[17]:

#This part of the pipeline is used for visualization and validation of prefered models

def roc_graph(X_data, y_data, model_):
    """
    Given feature and target data, creates an interactive ROC curve

    Parameters:
    X_data: Data to predict on
    Y_data: data to predict on
    Model: fitted model

    Returns:
        Prints ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_data, model_.predict_proba(X_data)[:,1])
    trace0 = go.Scatter(
    x = fpr,
    y = tpr,
    mode = 'lines',
    name = 'ROC curvce'
    )
    trace1 = go.Scatter(
        x = [0,1],
        y = [0,1],
        mode = 'lines',
        name = 'Reference Line'
    )
    data = [trace0, trace1]

    layout = dict(title = 'ROC (False Positives against True Positives)',
                  xaxis = dict(title = 'False Positive Rate'),
                  yaxis = dict(title = 'True Positive'),
                  )

    fig = dict(data=data, layout=layout)
    print("ROC AUC score = ", roc_auc_score(y_data, model_.predict_proba(X_data)[:,1]))
    iplot(fig)


# In[18]:


def precesion_recall_graph(X_data, y_data, model):
    '''
    Given feature and target data, creates an interactive precision and recall graph to help determine the optimal balance between each score. Also calls function to plot ROC curve
    
    Parameters:
        X_data: Data to predict on
        Y_data: data to predict on
        Model: fitted model
        
    Returns:
        precision_curve, recall_curve, threshold_curve
    '''

    precision_curve, recall_curve, threshold_curve =    precision_recall_curve(y_data, model.predict_proba(X_data)[:,1])

    trace0 = go.Scatter(
    x = threshold_curve,
    y = precision_curve[1:],
    mode = 'lines',
    name = 'precision'
    )
    trace1 = go.Scatter(
        x = threshold_curve,
        y = recall_curve[1:],
        mode = 'lines',
        name = 'recall'
    )
    data = [trace0, trace1]

    layout = dict(title = 'Precision vs. Recall',
                  xaxis = dict(title = 'Recall'),
                  yaxis = dict(title = 'Precision'),
                  )

    fig = dict(data=data, layout=layout)

    iplot(fig)
    
    roc_graph(X_data, y_data, model)
    
    return precision_curve, recall_curve, threshold_curve

