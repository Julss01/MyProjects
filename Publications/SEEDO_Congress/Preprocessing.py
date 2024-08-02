# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 12:03:06 2024

@author: julia
"""

import pandas as pd
import os
import numpy as np
import math as mt

def reading_files(path):
    dict_databases= {}
    for file in os.listdir(path): 
        dict_databases[file]=pd.read_csv(path+ str("/")+file, sep= ";",
                                         encoding='unicode_escape')
    return dict_databases
  
def changing_colnames(databases):
    list_new_cols= []
    dbs= {}
    for i in databases["2024_07_25_genobox_trabajo_julia.csv"]:
        i= i.rstrip()
        if " " in i: 
            i=i.replace(" ", "_")
        if "(" or ")" in i: 
            i=i.replace("(", "")
            i=i.replace(")", "")
        if "/" in i: 
            i= i.replace("/", "_")
        if i== "Peso_Kg":
            i=="Weight"
        list_new_cols.append(i)
    
    for db in databases: 
        dbs[db]=databases[db].set_axis(list_new_cols, axis=1)
    return dbs

def removing_NAval(databases):
    databases_no_NA= {}
    for db in databases:
        databases_no_NA[db]=databases[db].dropna()
    return databases_no_NA

def changing_age(ages):
    strings = np.array2string(ages)
    decimal_str= str(strings.split(".")[1])
    decimal= int(decimal_str)/12
    integer= strings.split(".")[0]
    decimal_age= round(int(integer) +  decimal, 2)
    return decimal_age
    
# def calculating_HOMA(bd):
    
def calculating_indexes(datos):
    datos= datos.reset_index(drop=True)
    altura= datos["Height"]
    peso= datos["Weight"]
    WC= datos["Perimetro_de_cintura"]
    HC= datos["Perimetro_de_cadera"]
    datos["BMI"]= round(peso /altura**2,2)
    datos["TMI"]= round(peso/altura**3,3)
    datos["WHR"]= round(WC/HC,2)
    datos["WHtR"]= round(WC/altura, 2)
    for index in range(len(datos["Sex"])):
        if datos.loc[index,"Sex"]==0:
            datos["Hip_Index"]= round(datos.loc[index,"Perimetro_de_cadera"]*(datos.loc[index,"Weight"]**(-2/5))*((datos.loc[index,"Height"]*100)**(1/5)),2)
            datos["RFM"]= round(64-(20* (datos.loc[index,"Height"]/datos.loc[index,"Perimetro_de_cintura"])),2)
        elif datos.loc[index, "Sex"]==1:
            datos["Hip_Index"]=round(datos.loc[index,"Perimetro_de_cadera"]*(datos.loc[index,"Weight"]**-0.482)*((datos.loc[index,"Height"]*100)**0.310),2)
            datos["RFM"]= round(76-(20* (datos.loc[index,"Height"]/datos.loc[index,"Perimetro_de_cintura"])),2)
    datos["ABSI"]=round((WC/100)*(peso**(2/3))*(altura**1/2),2)
    datos["BAI"]= round(HC/(altura**1.5)-18,2)
    datos["Conicidad"]=round((WC*100)/0.109*(mt.sqrt(peso/altura)),2) 
    datos["BRI"]= round(364.2-365.5*mt.sqrt(1-((WC/2*mt.pi)**2)),2) 
    datos["TyG"]= round(mt.log(datos["TAG_mg_dl"]*datos["Glucose_mg_dl"])/2,2)    
    datos["LAratio"]= round(datos["Leptin_ng_ml"]/datos["Adiponectin_ng_ml"])
    return datos
    
def processing_iberomics(iberomics):
    iberomics= iberomics.reset_index(drop=True)
    iberomics["Height"]= iberomics["Height"]/100
    for index,code in enumerate(iberomics["Code"]):
        if "OZ" in code:
            if code!= "OZ120" or code!= "OZ124":
                if np.array2string(iberomics.loc[index,"Age"]).split(".")[1] != "":
                    iberomics.loc[index, "Age"]=changing_age(iberomics.loc[index,"Age"])
    iberomics.rename(columns={"Adiponectin_mg_l": "Adiponectin_ng_ml", 
                              "Leptin_ug_l": "Leptin_ng_ml"}, inplace=True)
    for index,SBP in enumerate(iberomics["SBP"]):
        if iberomics.loc[index,"SBP"]< iberomics.loc[index, "DBP"]:
            print(f"Hay un valor fisiológicamente anormal en: {index}")
    iberomics["Sex"].replace({1:0, 2:1}, inplace= True)
    iberomics=calculating_indexes(iberomics)
    return iberomics 

def processing_genobox(genobox):
    genobox= genobox.reset_index(drop= True)
    for index, value in enumerate(genobox["Estadio_tanner"]):
        if value== 98:
            genobox.drop(index, axis= 'index', inplace= True)
    genobox= genobox.reset_index(drop= True)
    genobox.rename(columns={"Adiponectin_mg_l": "Adiponectin_ng_ml"}, inplace=True)
    genobox.rename(columns={"Leptin_ug_l": "Leptin_ng_ml"}, inplace=True)
    genobox["Perimetro_de_cintura"]=pd.to_numeric(genobox.loc[:,"Perimetro_de_cintura"])
    for index,SBP in enumerate(genobox["SBP"]):
        if genobox.loc[index,"SBP"]< genobox.loc[index, "DBP"]:
            print(f"Hay un valor fisiológicamente anormal en Genobox: {index}")
            print(f"SBP:{genobox.loc[index, 'SBP']} , DBP:{genobox.loc[index, 'DBP']}")
    genobox= calculating_indexes(genobox)
    return genobox 
        
def saving_CSV(dict_dbs):
    for db in dict_dbs:
        dict_dbs[db].to_csv("C:/Users//" 
                            + "/" + db + ".csv",sep= ",")
   
databases= reading_files("C:/Users//")
new_dbs= changing_colnames(databases) 
dbs_NA= removing_NAval(new_dbs)

dbs_final= {}
dbs_final["2024_07_25_iberomics"]= processing_iberomics(dbs_NA["2024_07_25_iberomics_trabajo_julia.csv"])
dbs_final["2024_07_29_genobox"]= processing_genobox(dbs_NA["2024_07_25_genobox_trabajo_julia.csv"])
saving_CSV(dbs_final)