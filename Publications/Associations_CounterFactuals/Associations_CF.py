# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 18:18:28 2024

@author: julia
"""

import numpy as np
import shap
import copy
import math
#from sklearn.datasets import load_iris
import csv
import pandas as pd
from datetime import datetime
# Abre el archivo en modo de lectura
import dice_ml
from dice_ml.utils import helpers # helper functions
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
from scipy.stats import zscore
#retocar esto
outValues=2

#Importing the model from R
##### 

dic_list=[]
for child in range(0, len(shap_interaction_values[1])):
    dic_interactions= {}
    for i in range(0, len(shap_interaction_values[1][child][:][0])):
        for j in range(i+1,len(shap_interaction_values[1][child][0][:])):
            value= shap_interaction_values[1][child][i][j]
            if value== 0:
                continue
            
            key= dataframe.columns[i]+":"+dataframe.columns[j]
            dic_interactions[key]= value
    dic_list.append(dic_interactions)
    
dic_list_sort= []

for dic in dic_list: 
    sort= dict(sorted(dic.items(), key=lambda x: x[1], reverse=True))
    dic.clear()
    dic.update(sort)
    dic_list_sort.append(dic)


lista_top= []
for dic in dic_list_sort:
    lista_keys= list(dic.keys())
    dic_top= {}
    for key in lista_keys[:10]:
        dic_top[key]= dic[key]
    for key in lista_keys[len(lista_keys)-10: len(lista_keys)]:
        dic_top[key]= dic[key]
    lista_top.append(dic_top)

lista=[]
c=0
for dic in lista_top:
    maxi= max(dic.values())
    mini= min(dic.values())
    for key in dic.keys():
        if dic[key]== maxi or dic[key]== mini:
            clave= key
            lista.append(key)
            print(f"Una interacción importante para el diccionario número {c} es {clave} y su valor es {dic[key]}")
    c= c+1
    
repes={}
for key in lista:
    if key not in repes:
        repes[key]= lista.count(key)

repes_t= sorted(repes.items(), key=lambda x:x[1], reverse=True)

data_T= pd.read_csv("C:/Users/julil/Desktop/Pruebas/data_obesity_T.csv", sep= ";", encoding='utf-8')
data_T["Class"].replace(['Yes_IR', 'No_IR'], [1, 0], inplace=True)   
def plots(feature1, feature2, valor):
    plt.figure(1)
    fig = plt.figure(tight_layout=True, figsize=(20,10))
    spec = gridspec.GridSpec(ncols=3, nrows=2, figure=fig)
    
    ax0 = fig.add_subplot(spec[0, 0])
    shap.dependence_plot(feature1, shap_values_mio[1], 
                         dataframe, display_features=dataframe, 
                         interaction_index=feature2, ax=ax0, show=False)
    ax0.yaxis.label.set_color('black')          #setting up Y-axis label color to blue
    ax0.xaxis.label.set_color('black')          #setting up Y-axis label color to blue
    ax0.tick_params(axis='x', colors='black')    #setting up X-axis tick color to red
    ax0.tick_params(axis='y', colors='black')    #setting up X-axis tick color to red
    ax0.set_title(f'SHAP main effect', fontsize=10)
    ax1 = fig.add_subplot(spec[0, 1])
    shap.dependence_plot((feature1, feature2), shap_interaction_values[1], dataframe, display_features=dataframe, ax=ax1, axis_color='black', show=False)
    ax1.yaxis.label.set_color('black')          
    ax1.xaxis.label.set_color('black')         
    ax1.tick_params(axis='x', colors='black')    
    ax1.tick_params(axis='y', colors='black')    
    ax1.set_title(f'SHAP interaction effect', fontsize=10)
    indices= data_T.loc[(data_T[feature2]>=valor)].index
    plt.figure(2)
    fig2 = plt.figure(tight_layout=True, figsize=(20,10))
    temp = pd.DataFrame({feature1: data_T.loc[(data_T[feature2]>=valor), feature1].values,
                      'Probability for prediction 1': predictions[indices][:,1]})
    temp = temp.sort_values(feature1)
    temp.reset_index(inplace=True)
    ax3= fig2.add_subplot(spec[0, 0])
    sns.scatterplot(x=temp[feature1], y=temp.iloc[:,2].rolling(10,center=True).mean(), data=temp, ax= ax3, 
                    s=2, size= 4, cmap= "Oranges")
    ax3.legend()
    ax3.set_title(f'How the target probability depends on {feature1} and {feature2} is more than {valor} (mean)', fontsize=10)
    plt.figure(3)
    fig3 = plt.figure(tight_layout=True, figsize=(20,10))
    ax4= fig3.add_subplot(spec[0, 0])
    sns.scatterplot(x=temp[feature1], y=temp.iloc[:,2], data=temp, ax= ax4, 
                    s=2, size= 4, c=data_T.loc[(data_T[feature2]>=valor, feature2)], cmap= "Oranges")
    ax4.set_title(f'How the target probability depends on {feature1} and {feature2} is more than {valor}', fontsize=10)
    plt.figure(4)
    fig4= plt.figure(tight_layout=True, figsize=(20,10))
    ax5 = fig4.add_subplot(spec[1, 0])
    sns.scatterplot(x=feature1, y=feature2, data=data_T, hue="Class", ax=ax5, s=2, size=4)
    ax5.set_title(f'scatter plot', fontsize=10)
plots("cg19194924","Iron_(ug/dl)",1)
plots("cg09109553_TOLLIP","Iron_(ug/dl)",1)
plots("cg18517961_SPNS2","Iron_(ug/dl)",1)
plots("Iron_(ug/dl)","cg02818143_PTPRN2",1)
plots("Iron_(ug/dl)","cg27147114_RASGRF1",1)
plots("cg03639328_CYTH3", "HDLc_(mg/dl)",1)
plots("cg10987850_HMCN1","HDLc_(mg/dl)",1)
plots("cg02818143_PTPRN2","HDLc_(mg/dl)",1)
plots("cg21024835_PTPRN2","BMI_zscore",1)
plots("cg01851968_VASN","BMI_zscore",1)
plots("cg08541862_ELOVL1","BMI_zscore",1)
plots("Sex", "cg00917561_CLASP1",1)
plots("Sex", "cg00152126_CTBP2",1)
plots("Sex", "cg03141724_PRKCQ",1)
plots("Sex", "cg17979173_TNXB",1)
plots("Sex", "cg00152126_CTBP2", 5.2)



#Interaction Graph

# mean_shap = (np.abs(shap_interaction_values[1]).mean(0))*10**3
# df = pd.DataFrame(mean_shap,index=dataframe.columns,columns=dataframe.columns)
# vars_heatmap= ["cg04976245_PTPRN2", "Adiponectin_leptin_ratio", "cg06267617_DNM3", "cg13153055_SOX6", "cg11762807_HDAC4",
#                 "cg10956605_VIPR2","cg08541862_ELOVL1", "BMI_zscore", "cg03516256_EBF1", "cg11562025_FAM107B", "cg15888803_CDC42BPB",
#                 "cg17279138_ABCG1", "cg16219124_CLPTM1L","cg18073874_EEFSEC", "cg10937973_CLASP1","cg07792979_MATN2", "cg23792592_MIR1-1",
#                 "Sex", "cg00917561_CLASP1", "cg00152126_CTBP2", "cg19139111_MATN2", "cg03141724_PRKCQ","cg02474195_SNRK",
#                 "cg17979173_TNXB", "cg27147114_RASGRF1", "cg02818143_PTPRN2"]

# df_v= df.loc[vars_heatmap, vars_heatmap]
# fig, ax= plt.subplots(figsize=(15,15))
# sns.set(font_scale=2)
# sns.heatmap(df_v,vmin=0, vmax= 0.8, cbar= True, fmt= ".1f", annot_kws= {"size": 10},
#                   annot=True, square= True, cmap= plt.cm.Blues, linewidths=0.5)
# ax.set_title("Heatmap Variables Importantes")
# plt.tight_layout()
# plt.savefig("heatmap_var_imp", dpi=300)
# fig, ax= plt.subplots(figsize=(30,30))
# variable= 50
# for i in range(int(300/variable)):
#     for j in range(int(300/variable)):
#         sns.heatmap(df.iloc[j*variable:(j+1)*variable,i*variable:(i+1)*variable,],vmin=0, vmax= 0.8, cbar= True, fmt= ".1f", annot_kws= {"size": 8},
#                   annot=True, square= True, cmap= plt.cm.Blues)
#         ax.set_title("Heatmap "+ str(i*(int(300/variable))+j))
#         plt.tight_layout()
#         plt.savefig("heatmap"+str(i*(int(300/variable))+j)+".png", dpi=300)
#         plt.clf()
#         print("Generado!")

#ax.set_xticklabels(df.columns, rotation=90, fontsize=5)
#ax.set_yticklabels(df.columns, rotation=360, fontsize=5)



# Counter-Factuals
data_T= pd.read_csv("C:/Users/julil/Desktop/Pruebas/data_obesity_T.csv", sep= ";", encoding='utf-8')  
data_T["Class"].replace(['Yes_IR', 'No_IR'],
                        [1, 0], inplace=True)   

data_T= data_T.drop(["Origin", "Sex"], axis=1)
data_T_2= data_T.drop(["Class"], axis= 1)
list_col= list(data_T_2.columns.values)

def display_df(df, show_only_changes, test_instance_df):
        from IPython.display import display
        if show_only_changes is False:
            display(df)  # works only in Jupyter notebook
        else:
            newdf = df.values.tolist()
            org = test_instance_df.values.tolist()[0]
            for ix in range(df.shape[0]):
                for jx in range(len(org)):
                    if not isinstance(newdf[ix][jx], str):
                        if math.isclose(newdf[ix][jx], org[jx], rel_tol=abs(org[jx]/10000)):
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
                    else:
                        if newdf[ix][jx] == org[jx]:
                            newdf[ix][jx] = '-'
                        else:
                            newdf[ix][jx] = str(newdf[ix][jx])
            return pd.DataFrame(newdf, columns=df.columns, index=df.index)


d= dice_ml.Data(dataframe=data_T,continuous_features= list_col, outcome_name= "Class")
m = dice_ml.Model(model=tree_exp.model, backend="sklearn")
exp = dice_ml.Dice(d, m, method="random")

list_indiv= ["S69608", "S75679", "R24959", "R39751", "Z091_1", "S85555", "S58671","S825037", "R94897", "S41894" ]

df_ind= pd.DataFrame()
for i in list_indiv:
    e= exp.generate_counterfactuals(data_T.drop("Class", axis=1)[dataframe.index == i], total_CFs=1, desired_class="opposite", initial_predictions= predictions[dataframe.index == i])
    df_i= display_df(e.cf_examples_list[0].final_cfs_df_sparse, True, e.cf_examples_list[0].test_instance_df)
    df_i.insert(0, "ID",[i], allow_duplicates=True)
    df_ind= pd.concat([df_i, df_ind])

df_ind.to_csv("C:/Users/julil/Desktop/Pruebas/df_counterfactuals.csv", index= False)
       