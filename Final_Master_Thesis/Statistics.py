# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 23:50:46 2024

@author: julil
"""

import pandas as pd
from scipy.stats import shapiro
from scipy.stats import levene, ttest_ind
import os

def open_df(folder_path):
    DBs= os.listdir(folder_path)
    dbs_dict={}
    for db in DBs:
        dbs_dict[db.split(".")[0]]= pd.read_csv(folder_path + "/" +str(db), sep= ";")
    for db_name in dbs_dict.keys():
        dbs_dict[db_name] = dbs_dict[db_name][dbs_dict[db_name].columns.drop(list(dbs_dict[db_name].filter(regex='Unnamed')))]
    return dbs_dict

#Homogeneidad de varianzas
dbs= open_df("C:/Users/julil/Desktop/TFM/Disp")

# levene_stat_ig, levene_p_ig = levene(dbs["iberomics_F_dxa_tronco_processed"]["homa_zscore_stavnsbo"]
#                                      ,dbs["pubmep_F_dxa_tronco_processed"]["homa_zscore_stavnsbo"])
# levene_stat_im, levene_p_im = levene(dbs["iberomics_M_dxa_tronco_processed"]["homa_zscore_stavnsbo"],
#                                      dbs["pubmep_M_dxa_tronco_processed"]["homa_zscore_stavnsbo"])



# #Normality
sg,pg= shapiro([dbs["iberomics_F_dxa_tronco_processed"]["decimal_age"]]) 
sm,pm= shapiro([dbs["iberomics_M_dxa_tronco_processed"]["decimal_age"]])

sgp, pgp=  shapiro([dbs["pubmep_F_dxa_tronco_processed"]["decimal_age"]]) 
smp, pmp=  shapiro([dbs["pubmep_M_dxa_tronco_processed"]["decimal_age"]]) 

#t de student

tstatG, tpG= ttest_ind(dbs["iberomics_F_dxa_tronco_processed"]["decimal_age"], 
                      dbs["pubmep_F_dxa_tronco_processed"]["decimal_age"])

tstatM, tpM= ttest_ind(dbs["iberomics_F_dxa_tronco_processed"]["decimal_age"], 
                      dbs["pubmep_M_dxa_tronco_processed"]["decimal_age"])
