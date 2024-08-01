# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 16:56:59 2024

@author: julil
"""

import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from cubist import Cubist
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
        y=dictionary_df[db_name].pop("homa_zscore_stavnsbo")
        x=dictionary_df[db_name].drop("code", axis=1)
        dict_cols[db_name + "_columns"]= dictionary_df[db_name].columns
        if "iberomics" in db_name:
            numpy_dict[db_name + "_TrainX"]=x
            numpy_dict[db_name + "_TrainY"]=y
        else:
            test_dict[db_name + "_TestX"]= x
            test_dict[db_name + "_TestY"]=y
    return numpy_dict, dict_cols, test_dict

def opt(trainX_Y):
    dict_comp= {}
    clf_EN= {}
    params= {}
    c=0
    M5= Cubist()
    ss= StandardScaler()
    pipe=Pipeline(steps=[("M5", M5)])
    for trainX in trainX_Y.keys(): 
        if "_TrainX" in trainX: 
            # n_samples, n_features= trainX_Y[trainX].shape
            n_committees= [1, 10, 1] 
            neighbors = [1, 9, 1]
            max_iter= [200000]
            dict_comp[trainX]= dict(M5__n_committees= n_committees,
                                    M5__neighbors= neighbors) 
    
    for X_Y in trainX_Y.keys():
        if "TrainX" in X_Y:
            scaled=ss.fit_transform(trainX_Y[X_Y])
            x= scaled
            # for var in var_out_dict.keys():
            #     if "age" in var and var.split("age")[1]== X_Y.split("_TrainX")[0]:
            #         ag=var_out_dict[var].to_numpy()
            #         np.c_[x,ag]
            #     elif "tanner" in var and var.split("tanner")[1]== X_Y.split("_TrainX")[0]:
            #         tan=var_out_dict[var]
            #         np.c_[x, tan]
            clf_EN = GridSearchCV(pipe, dict_comp[X_Y])
            c+=1
        elif "TrainY" in X_Y:
            y= trainX_Y[X_Y]
            c+=1
        if c==2: 
            clf_EN.fit(x,y)
            best_params = clf_EN.best_estimator_.get_params()
            b = best_params["M5"]
            print(f"Para el modelo {X_Y} el mejor estimador es: \n{b}")
            params[X_Y.split("_Train")[0]] = b
            c=0
    return params

# def model(dbs_dict):
#     model_dict= {}
#     c= 0
#     ss= StandardScaler()
#     for db_name in dbs_dict.keys():
#         if "TrainX" in db_name:
#             x= dbs_dict[db_name]
#             scaled_x= ss.fit_transform(x)
#             # nx= normalize(x)
#             c+=1
#         elif "TrainY" in db_name:
#             y= dbs_dict[db_name]
#             c+=1
#         if c==2:
#             model= cubist()
#             model_dict[db_name.split("_Train")[0]]= model.fit(scaled_x,y)
#             c=0
#     return model_dict

def predict(model_dict, test_dict):
    predictions= {}
    ss= StandardScaler()
    for model in model_dict.keys():
        model1= model_dict[model]
        for test in test_dict.keys():
            if model.split("iberomics_")[1] in test and "TestX" in test:
                X= test_dict[test]
                scaled_X= ss.fit_transform(X)
                # nX= normalize(X)
                predictions[test.split("_Test")[0]]= model1.predict(scaled_X)
    return predictions

def plot(predictions, testY):
    for pred_name in predictions: 
        y= testY[pred_name + "_TestY"]
        fig= plt.figure()
        plt.title(pred_name + " VS " + pred_name + "_TestY")
        plt.xlabel("Sample")
        plt.ylabel("Values")
        plt.scatter(np.arange(len(predictions[pred_name])),predictions[pred_name], c="lightblue", label= "Prediction")
        # plt.plot(np.arange(len(predictions[pred_name])),predictions[pred_name], color="lightblue")
        plt.scatter(np.arange(len(y)),y, c="lightpink", label= "True Values")
        # plt.plot(np.arange(len(y)),y, color="lightpink")
        plt.legend()
        plt.ylim([-4,5])
        
def plot_sample(numpy_dict, test_dict, predictions):
    for model in numpy_dict:
        if "TrainY" in model:
            y_train= numpy_dict[model]
            y_test= test_dict["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]+ "_TestY"]
            # y_pred= predictions["pubmep_"+model.split("iberomics_")[1].split("_Train")[0]]
            plt.figure()
            plt.title(model.split("iberomics_")[1].split("_Train")[0])
            plt.xlabel("Sample")
            plt.ylabel("Values")
            plt.scatter(np.arange(len(y_train)),y_train, c= "lightblue", label= "Cohorte Entrenamiento" )
            plt.scatter(np.arange(len(y_test)),y_test, c= "lightpink", label= "Cohorte Validación" )
            # plt.scatter(np.arange(len(y_pred)),y_pred, c= "lightgreen", label= "Predicción" )
            plt.legend()
            plt.ylim([-4,5])


def model_analysis(test_Y, predictions):
    dict_RMSE= {}
    dict_r2= {}
    c=0
    for db_name in test_Y: 
        if "_TestX" in db_name:
            c=c+1
            testX= test_Y[db_name]
        elif "_TestY" in db_name: 
            testY= test_Y[db_name]
            for pred in predictions: 
                if db_name.split("_TestY")[0]==pred:
                    testY= test_Y[db_name]
                    predict= predictions[db_name.split("_TestY")[0]]
                    dict_RMSE[db_name + "_RMSE"]= np.sqrt(mean_squared_error(testY, predict))
                    dict_r2[db_name + "_r2"]= r2_score(testY, predict)
    return dict_RMSE, dict_r2

def coef(model_dict):
    for model in model_dict:
        if model.split("iberomics_final_")[1]== "F_processed":
            imp_F=params["iberomics_final_F_processed"].feature_importances_
            use_F= params["iberomics_final_F_processed"].coeff_
            t_F= params["iberomics_final_F_processed"].rules_
        elif model.split("iberomics_final_")[1]== "M_processed":
            imp_M=params["iberomics_final_F_processed"].feature_importances_
            use_M= params["iberomics_final_F_processed"].coeff_
            t_M= params["iberomics_final_F_processed"].rules_
    return imp_F, use_F, t_F, imp_M, use_M, t_F
    
# #MAIN
dbs_dict= open_df("C:/Users/julil/Desktop/TFM/BDs_pasar_M5")
numpy_dict, dict_cols, test_dict= transform_into_numpy(dbs_dict)
params=opt(numpy_dict)
# model_dict= model(numpy_dict)
pred= predict(params, test_dict)
RMSE, r2= model_analysis(test_dict, pred)
plot(pred, test_dict)
plot_sample(numpy_dict, test_dict, pred)

iF, uF, tF, iM, uM, tM= coef(params)






