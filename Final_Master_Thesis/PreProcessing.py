# -*- coding: utf-8 -*-
"""
Created on Mon May 27 19:17:39 2024

@author: julil
"""

import pandas as pd
import numpy as np
import os
from sklearn.covariance import EllipticEnvelope
import matplotlib.pyplot as plt
import pingouin as pg

def open_df(folder_path):
    DBs= os.listdir(folder_path)
    dbs_dict={}
    for db in DBs:
        dbs_dict[db.split(".")[0]]= pd.read_csv(folder_path + "/" +str(db), sep= ";")
    return dbs_dict

def unify_db(databases):
    for db_name in databases.keys():
        if "genobox" in db_name:
            pubmep = [i for i in databases.keys() if "pubmep" in i]
            merged= pd.DataFrame.merge(databases[pubmep[0]], databases[db_name], on="code")

            new_genobox= pd.DataFrame()
            for i in merged["code"].index:
                databases[db_name]=databases[db_name].replace(merged.loc[i,"code"],merged.loc[i,"code_new_t2"])
                new_genobox=pd.concat([new_genobox,databases[db_name].loc[databases[db_name]["code"]==merged.loc[i,"code_new_t2"]]], axis=0)

            databases[db_name]=new_genobox

    keys=list(databases.keys())      
    for db_name in keys:
        db=databases[db_name]
        if "dexa_pubmep" in db_name: 
            db.drop("code", inplace= True, axis=1)
            db.rename(columns= {"code_new_t2":"code"}, inplace= True)
            for names in db.columns:
                db.rename(columns={names:names.split("_t2")[0]},inplace= True)
                db.rename(columns={names:names.split("_t1")[0]},inplace= True)
            db.replace("Varon_T2", 0, inplace= True)  
            db.replace("Niña_T2", 1, inplace= True) 
        if "dexa_iberomics" in db_name:
            db["sex"].replace(1, 0, inplace= True)  
            db["sex"].replace(2, 1, inplace= True) 
        if "score_pubmep" in db_name:
            db.drop("code",inplace= True, axis=1)
            db.rename(columns= {"code_new_t2":"code"}, inplace= True)
            databases["pubmep_t1_bioq"]=db.loc[:,~db.columns.str.endswith('_t2')]
            databases["pubmep_t2_bioq"]=db.loc[:,~db.columns.str.endswith('_t1')]
            
def preprocessing(databases, variables):
    for db_name in databases.keys():
        if "dexa" in db_name:
            base=databases[db_name]
            base=base[variables]
            base=base.dropna()
            
            if base.iloc[0]["height"] > 50:
                base["height"]= base["height"]/100

            base["mgrasatg"]= base["mgrasatg"]/1000
            base["mmagratg"]= base["mmagratg"]/1000
            base["tronmgrasa"]= base["tronmgrasa"]/1000
            base["bramgrasa"] = base["bramgrasa"] /1000
            base["piemgrasa"]= base["piemgrasa"] / 1000
            base["tronmmagra"]= base["tronmmagra"]/1000
            base["brammagra"] = base["brammagra"] /1000
            base["piemmagra"]= base["piemmagra"] / 1000
            
            lmi_fmi= lambda a,b : a/(b**2)
            
            base["lmi_tan"]= lmi_fmi(base['mg_kg_tan'],base["height"])
            base["lmi_dxa"]= lmi_fmi(base["mmagratg"],base["height"])
            base["fmi_tan"]= lmi_fmi(base['mm_kg_tan'],base["height"])
            base["fmi_dxa"]= lmi_fmi(base["mgrasatg"],base["height"])
            databases[db_name]=base
            
        elif "bioq" in db_name: 
            base=databases[db_name]
            base=base.dropna()
            databases[db_name]=base
            
def outliers(tan, DXA, c):   
    data = np.column_stack((tan.values, DXA.values))
    mcd= EllipticEnvelope(contamination=c) #Changing contamination according to the outliers % expected
    mcd_tan= mcd.fit(data)                                                                                                                          
    outliers= mcd_tan.predict(data) == -1  
    return outliers

def plot_outliers(database, x_lab,y_lab,title,c, var1, var2=None, sum_db=None):
    plt.figure()
    plt.xlabel(x_lab)
    plt.ylabel(y_lab)
    plt.title(title)
    if sum_db:
        df= pd.concat([database[sum_db[0]], database[sum_db[1]], database[sum_db[2]]], axis=1)
        var2= df.sum(axis=1)
    plt.scatter(var1,var2, cmap= 'Oranges')
    outliers_values= outliers(var1, var2, c)
    for i in var1[outliers_values].index: 
        ID= database["code"].loc[i]
        plt.annotate(ID, (var1[i], var2[i]))
    corr= pg.corr(var1,var2, method= "pearson")
    plt.text(max(var1)+5, min(var2), "r: "+ str(corr.iloc[0,1]) + "\np-value: " + str(corr.iloc[0,3]))

def exporting_db(database, folder_path):
    for db_name in database.keys():
        database[db_name].to_csv(folder_path + "/" + str(db_name)+ "_processed.csv")

# MAIN

databases= open_df("C://")

unify_db(databases)
        
var= ["code", "height", "sex", 'mg_kg_tan', 'mm_kg_tan', 
      'mgrasatg', 'mmagratg',  'tronmmagra',
      'tronmgrasa','brammagra','bramgrasa', 
      'piemmagra','piemgrasa']  

preprocessing(databases, var)

for db_name in databases.keys():
    if "dexa" in db_name: 
        plot_outliers(databases[db_name], "Masa Grasa Tan",
                      "Masa Grasa DXA", "Masa Grasa Tan vs DXA "+ str(db_name), 0.01,
                      databases[db_name]["mg_kg_tan"],
                      databases[db_name]["mgrasatg"])    
        plot_outliers(databases[db_name], "Masa Magra Tan",
                      "Masa Magra DXA", "Masa Magra Tan vs DXA "+ str(db_name), 0.01,
                      databases[db_name]["mm_kg_tan"],
                      databases[db_name]["mmagratg"])   
        plot_outliers(databases[db_name], "Masa Grasa DXA",
                      "Masa Grasa Suma DXA", "Masa Grasa DXA vs DXA suma "+ str(db_name), 0.01,
                      databases[db_name]["mgrasatg"],
                      sum_db= ["tronmgrasa", "bramgrasa", "piemgrasa"])   
        plot_outliers(databases[db_name], "Masa Magra DXA",
                      "Masa Magra Suma DXA", "Masa Magra DXA vs DXA suma "+ str(db_name), 0.01,
                      databases[db_name]["mmagratg"],
                      sum_db=["tronmmagra", "brammagra", "piemmagra"])   
    
exporting_db(databases, "C:/Users//")
        