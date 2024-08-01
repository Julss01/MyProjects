# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 22:59:36 2024

@author: julil
"""

from sklearn.linear_model import ElasticNet, ElasticNetCV
import os
import pandas as pd
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn import decomposition
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def open_df(folder_path):
    DBs= os.listdir(folder_path)
    dbs_dict={}
    for db in DBs:
        dbs_dict[db.split(".")[0]]= pd.read_csv(folder_path + "/" +str(db), sep= ";")
    for db_name in dbs_dict.keys():
        dbs_dict[db_name] = dbs_dict[db_name][dbs_dict[db_name].columns.drop(list(dbs_dict[db_name].filter(regex='Unnamed')))]
    return dbs_dict


def transform_into_numpy(dictionary_df):
    numpy_dict= {}
    dict_cols= {}
    test_dict= {}
    for db_name in dictionary_df.keys():
        y=dictionary_df[db_name].pop("homa_zscore_stavnsbo").to_numpy()
        x=dictionary_df[db_name].drop("code", axis=1).to_numpy()
        dict_cols[db_name + "_columns"]= dictionary_df[db_name].columns
        if "iberomics" in db_name:
            numpy_dict[db_name + "_TrainX"]=x
            numpy_dict[db_name + "_TrainY"]=y
        else:
            test_dict[db_name + "_TestX"]= x
            test_dict[db_name + "_TestY"]=y
    return numpy_dict, dict_cols, test_dict

def selecting_variables(trainX_Y,test_dict,dict_cols):
    dict_var_train= {}
    dict_var_names= {}
    dict_var_test= {}
    c=0
    n_var=0
    for model in trainX_Y.keys():
        if "TrainX" in model:
            n_var=trainX_Y[model].shape[1]
            if n_var>20:
                rfe = RFE(estimator=ElasticNet(),n_features_to_select=50)
                x=trainX_Y[model]
                c+=1
            else:
                dict_var_train[model]=trainX_Y[model]
                dict_var_test["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]=test_dict["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]
                # dict_var_test["pubmep_t2_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]=test_dict["pubmep_t2_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]
        elif "TrainY" in model:
            dict_var_train[model]=trainX_Y[model]
            dict_var_test["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestY"]=test_dict["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestY"]
            # dict_var_test["pubmep_t2_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestY"]=test_dict["pubmep_t2_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestY"]
            if n_var>20:
                y=trainX_Y[model]
                c+=1
        if c==2:
            rfe.fit(x,y)
            col_list=[]
            for index,elemnt in enumerate(rfe.support_.tolist()):
                if elemnt:
                    col_list.append(dict_cols[model.split("_Train")[0]+"_columns"][index])
            dict_var_names[model.split("_Train")[0]]=col_list
            array_train=np.zeros((len(trainX_Y[model.split("_Train")[0]+"_TrainX"]),50))
            array_test_t1=np.zeros((len(test_dict["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]),50))
            # array_test_t2=np.zeros((len(test_dict["pubmep_t2_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]),50))
            c2=0
            for index,element in enumerate(rfe.support_.tolist()):
                if element:
                    array_train[:,c2]=trainX_Y[model.split("_Train")[0]+"_TrainX"][:,index]
                    array_test_t1[:,c2]=test_dict["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"][:,index]
                    # array_test_t2[:,c2]=test_dict["pubmep_t2-"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"][:,index]
                    c2+=1
            dict_var_train[model.split("_Train")[0]+"_TrainX"]=array_train
            dict_var_test["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]=array_test_t1
            # dict_var_test["pubmep_t2-"+model.split("iberomics_")[1].split("_Train")[0]+"_TestX"]=array_test_t2
            c=0
            
            
    return dict_var_train,dict_var_names,dict_var_test


def opt(trainX_Y):
    dict_comp= {}
    clf_EN= {}
    params= {}
    c=0
    elasticnet= ElasticNet()
    # stc_slc= StandardScaler()
    pipe=Pipeline(steps=[("elasticnet", elasticnet)])
    for trainX in trainX_Y.keys(): 
        if "_TrainX" in trainX: 
            # n_samples, n_features= trainX_Y[trainX].shape
            alpha= np.arange(0.001,12,0.1)
            l1_ratio= [0.1, 0.2, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
            selection = ['cyclic', 'random']
            max_iter= [200000]
            dict_comp[trainX]= dict(elasticnet__alpha= alpha,
                                    elasticnet__max_iter= max_iter,
                                    elasticnet__l1_ratio= l1_ratio, 
                                    elasticnet__selection= selection) 
    
    for X_Y in trainX_Y.keys():
        if "TrainX" in X_Y:
            x= trainX_Y[X_Y]
            clf_EN = GridSearchCV(pipe, dict_comp[X_Y])
            c+=1
        elif "TrainY" in X_Y:
            y= trainX_Y[X_Y]
            c+=1
        if c==2: 
            clf_EN.fit(x,y)
            best_params = clf_EN.best_estimator_.get_params()
            b = best_params["elasticnet"]
            print(f"Para el modelo {X_Y} el mejor estimador es: \n{b}")
            params[X_Y.split("_Train")[0]] = b
            c=0
    return params



def model(list_vars):
    model_dict= {}
    coef_dict= {}
    intercept_dict= {}
    
    for index in range(len(list_vars)):
        x= list_vars[index][1]
        y= list_vars[index][2]
        # for db_alpha in alpha.keys():
        #     if db_alpha.split("alpha_")[1] in list_vars[index][0]: 
        model= ElasticNet() #alpha= alpha[db_alpha]
        model_dict[list_vars[index][0]]= model.fit(x,y)
        coef_dict[list_vars[index][0]]= model.coef_
        intercept_dict[list_vars[index][0]]= model.intercept_
    return model_dict, coef_dict, intercept_dict

def predict(model_dict, var_test):
    predictions= {}
    for model in model_dict.keys():
        model1= model_dict[model]
        for test in var_test.keys():
            if model.split("iberomics_")[1]==test.split("pubmep_")[1].split("_Test")[0]:
                if "TestX" in test:
                    X= var_test[test]
                    predictions[test.split("_Test")[0]]= model1.predict(X)
    return predictions

def model_analysis(test_Y, predictions):
    dict_RMSE= {}
    dict_r2= {}
    for db_name in test_Y: 
        if "_TestY" in db_name: 
            for pred in predictions: 
                if db_name.split("_TestY")[0]==pred:
                    testY= test_Y[db_name]
                    predict= predictions[db_name.split("_TestY")[0]]
                    dict_RMSE[db_name + "_RMSE"]= np.sqrt(mean_squared_error(testY, predict))
                    dict_r2[db_name + "_r2"]= r2_score(testY, predict)
    return dict_RMSE, dict_r2

def plot(predictions, var_test):
    for test in var_test.keys():
        if "TestY" in test:
            y= var_test[test]
            pred_name= test.split("_TestY")[0]
            fig= plt.figure()
            plt.title(pred_name + " VS " + pred_name + "_TestY")
            plt.xlabel("Sample")
            plt.ylabel("Values")
            plt.scatter(np.arange(len(predictions[pred_name])),predictions[pred_name], c="lightblue", label= "Prediction")
            plt.plot(np.arange(len(predictions[pred_name])),predictions[pred_name], color="lightblue")
            plt.scatter(np.arange(len(y)),y, c="lightpink", label= "True Values")
            plt.plot(np.arange(len(y)),y, color="lightpink")
            plt.legend()
            plt.ylim([-4,5])
        
def forest_plot(model_dict, predictions, testY, coef_dict):
    models= []
    for model in model_dict.keys():
        if model.split("iberomics_")[1] in predictions and (model.split("iberomics_")[1]+ "_TestY") in testY:
            models.append(dict(modelo=model_dict[model],
                            coef= coef_dict[model],
                            real_val= testY[model.split("iberomics_")[1]+ "_TestY"], 
                            pred= predictions[model.split("iberomics_")[1]]))
    # for model in models:
    #     name = model['name']
    #     coef = model['coef']
    #     pred_values = model['pred_values']
    #     real_values = model['real_values']
    #     fig, ax = plt.subplots(figsize=(10, 6))
    #     variables = list(coef.keys())
     
    #     for i, var in enumerate(variables):
    #         # Dibujar la estimación puntual y el intervalo de confianza
    #         point = coef[var] 
    #         ax.plot([ci_low, ci_high], [i, i], color='black')
    #         ax.plot(point, i, 'o', color='red')
            
                
                
        
        
#MAIN
dbs_dict= open_df("C:/Users/julil/Desktop/TFM/BDs_pruebas")
numpy_dict, dict_cols, test_dict= transform_into_numpy(dbs_dict)
var_train,var_train_name,var_test=selecting_variables(numpy_dict,test_dict,dict_cols)
params= opt(var_train)
# model_dict, coef, intercept= model(list_var)
pred= predict(params, var_test)
RMSE, r2= model_analysis(test_dict, pred)
plot(pred, var_test)