# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 16:24:32 2024

@author: julil
"""

import pandas as pd
import numpy as np
import os

def open_df(folder_path):
    DBs= os.listdir(folder_path)
    dbs_dict={}
    for db in DBs:
        dbs_dict[db.split(".")[0]]= pd.read_csv(folder_path + "/" +str(db), sep= ";")
    return dbs_dict

def merging(databases, fmi_lmi_tani, fmi_lmi_dxai, tronco, brazo, piernas):
    final_dict= {}

    iberomics_final_tan= pd.DataFrame.merge(databases["2024_04_01_tanita_dexa_iberomics_processed"][fmi_lmi_tani], 
                                        databases["2024_05_07_bioquimica_homazscore_iberomics_processed"],
                                        on= "code")
    
    iberomics_final_dxa= pd.DataFrame.merge(databases["2024_04_01_tanita_dexa_iberomics_processed"][fmi_lmi_dxai], 
                                        databases["2024_05_07_bioquimica_homazscore_iberomics_processed"],
                                        on= "code")
    
    iberomics_final_dxat= pd.DataFrame.merge(databases["2024_04_01_tanita_dexa_iberomics_processed"][tronco], 
                                        databases["2024_05_07_bioquimica_homazscore_iberomics_processed"],
                                        on= "code")
    
    iberomics_final_dxab= pd.DataFrame.merge(databases["2024_04_01_tanita_dexa_iberomics_processed"][brazo], 
                                        databases["2024_05_07_bioquimica_homazscore_iberomics_processed"],
                                        on= "code")
    
    iberomics_final_dxap= pd.DataFrame.merge(databases["2024_04_01_tanita_dexa_iberomics_processed"][piernas], 
                                        databases["2024_05_07_bioquimica_homazscore_iberomics_processed"],
                                        on= "code")
    
    base= databases["2024_05_07_bioquimica_homazscore_iberomics_processed"][databases["2024_05_07_bioquimica_homazscore_iberomics_processed"]["code"].isin(iberomics_final_tan["code"])]
    final_dict["iberomics_F_base"]= base[base["sex"]==0]
    final_dict["iberomics_M_base"]= base[base["sex"]==1]
    
    final_dict["iberomics_F_fmi_lmi_tan"]= iberomics_final_tan[iberomics_final_tan["sex"]==0]
    final_dict["iberomics_M_fmi_lmi_tan"]= iberomics_final_tan[iberomics_final_tan["sex"]==1]
    
    final_dict["iberomics_F_fmi_lmi_dxa"]= iberomics_final_dxa[iberomics_final_dxa["sex"]==0]
    final_dict["iberomics_M_fmi_lmi_dxa"]= iberomics_final_dxa[iberomics_final_dxa["sex"]==1]
    
    final_dict["iberomics_F_dxa_tronco"]=  iberomics_final_dxat[iberomics_final_dxat["sex"]==0]
    final_dict["iberomics_M_dxa_tronco"]= iberomics_final_dxat[iberomics_final_dxat["sex"]==1]
    
    final_dict["iberomics_F_dxa_brazos"]=  iberomics_final_dxab[iberomics_final_dxab["sex"]==0]
    final_dict["iberomics_M_dxa_brazos"]= iberomics_final_dxab[iberomics_final_dxab["sex"]==1]
    
    final_dict["iberomics_F_dxa_piernas"]=  iberomics_final_dxap[iberomics_final_dxap["sex"]==0]
    final_dict["iberomics_M_dxa_piernas"]= iberomics_final_dxap[iberomics_final_dxap["sex"]==1]
    
    
    # final_dict["iberomics_F_fmi_lmi_tan_complete"]= pd.DataFrame.merge(databases["iberomics_F"]["mejorDB"], 
    #                                                   databases["scores_IBEROMICS"], on="code")
    # final_dict["iberomics_M_fmi_lmi_tan_complete"]= pd.DataFrame.merge(databases["iberomics_M"]["mejorDB"], 
    #                                                   databases["scores_IBEROMICS"], on="code")
    
    
    db=databases["2024_05_07_bioquimica_homazscore_pubmep_processed"]
    bio_t1=db.loc[:,~db.columns.str.endswith('_t2')]
    bio_t2=db.loc[:,~db.columns.str.endswith('_t1')].drop("code_new_t2",axis=1)
    
    for names in bio_t2.columns:
        bio_t2.rename(columns={names:names.split("_t2")[0]},inplace= True)
    for names in bio_t1.columns:
        bio_t1.rename(columns={names:names.split("_t1")[0]},inplace= True)    
    merged_tan1=pd.DataFrame.merge(databases["2024_04_24_tanita_dexa_genobox_processed"][fmi_lmi_tani],bio_t1,on="code")
    merged_tan2=pd.DataFrame.merge(databases["2024_03_21_tanita_dexa_pubmep_processed"][fmi_lmi_tani],bio_t2,on="code")
   
    merged_dxa1=pd.DataFrame.merge(databases["2024_04_24_tanita_dexa_genobox_processed"][fmi_lmi_dxai],bio_t1,on="code")
    merged_dxa2=pd.DataFrame.merge(databases["2024_03_21_tanita_dexa_pubmep_processed"][fmi_lmi_dxai],bio_t2,on="code")
    
    merged_t1=pd.DataFrame.merge(databases["2024_04_24_tanita_dexa_genobox_processed"][tronco],bio_t1,on="code")
    merged_t2=pd.DataFrame.merge(databases["2024_03_21_tanita_dexa_pubmep_processed"][tronco],bio_t2,on="code")
   
    merged_b1=pd.DataFrame.merge(databases["2024_04_24_tanita_dexa_genobox_processed"][brazo],bio_t1,on="code")
    merged_b2=pd.DataFrame.merge(databases["2024_03_21_tanita_dexa_pubmep_processed"][brazo],bio_t2,on="code") 
    
    merged_p1=pd.DataFrame.merge(databases["2024_04_24_tanita_dexa_genobox_processed"][piernas],bio_t1,on="code")
    merged_p2=pd.DataFrame.merge(databases["2024_03_21_tanita_dexa_pubmep_processed"][piernas],bio_t2,on="code")
    
    pubmep_tan=pd.concat([merged_tan1,merged_tan2],axis=0)
    pubmep_dxa=pd.concat([merged_dxa1,merged_dxa2],axis=0)
    pubmep_tronco=pd.concat([merged_t1,merged_t2],axis=0)
    pubmep_brazo= pd.concat([merged_b1,merged_b2],axis=0)
    pubmep_pierna= pd.concat([merged_p1,merged_p2],axis=0)
    
    base_v= pubmep_tan.drop(["fmi_tan","lmi_tan"], axis=1)
    final_dict["pubmep_F_base"]= base_v[base_v["sex"]==0].dropna()
    final_dict["pubmep_M_base"]= base_v[base_v["sex"]==1].dropna()
    
    final_dict["pubmep_F_fmi_lmi_tan"]= pubmep_tan[pubmep_tan["sex"]==0].dropna()
    final_dict["pubmep_M_fmi_lmi_tan"]= pubmep_tan[pubmep_tan["sex"]==1].dropna()
    
    final_dict["pubmep_F_fmi_lmi_dxa"]= pubmep_dxa[pubmep_dxa["sex"]==0].dropna()
    final_dict["pubmep_M_fmi_lmi_dxa"]= pubmep_dxa[pubmep_dxa["sex"]==1].dropna()
    
    final_dict["pubmep_F_dxa_tronco"]=  pubmep_tronco[pubmep_dxa["sex"]==0].dropna()
    final_dict["pubmep_M_dxa_tronco"]= pubmep_tronco[pubmep_dxa["sex"]==1].dropna()
    
    final_dict["pubmep_F_dxa_brazos"]=  pubmep_brazo[pubmep_dxa["sex"]==0].dropna()
    final_dict["pubmep_M_dxa_brazos"]= pubmep_brazo[pubmep_dxa["sex"]==1].dropna()
    
    final_dict["pubmep_F_dxa_piernas"]=  pubmep_pierna[pubmep_dxa["sex"]==0].dropna()
    final_dict["pubmep_M_dxa_piernas"]= pubmep_pierna[pubmep_dxa["sex"]==1].dropna()
    
    
    # final_dict["pubmep_F_fmi_lmi_tan_complete"]= pd.DataFrame.merge(databases["pubmep_F"]["mejorDB"], 
    #                                                   databases["scores_IBEROMICS"], on="code")
    # final_dict["pubmep_M_fmi_lmi_tan_complete"]= pd.DataFrame.merge(databases["pubmep_M"]["mejorDB"], 
    #                                                   databases["pubmep_IBEROMICS"], on="code")
    
    
    # databases["pubmep_t1_general"]= pd.DataFrame.merge(databases["2024_04_24_tanita_dexa_genobox_processed"],
    #                                       PGSp_t1_bioqp, on="code")
    # databases["pubmep_t2_general"]= pd.DataFrame.merge(databases["2024_03_21_tanita_dexa_pubmep_processed"],
    #                                       PGSp_t2_bioqp, on="code")
    
    
    # final_dict["pubmep_t1-fmi_lmi_tan"]= databases["pubmep_t1_general"][fmi_lmi_tanp1]
    # final_dict["pubmep_t2-fmi_lmi_tan"]= databases["pubmep_t2_general"][fmi_lmi_tanp2]
    
    # final_dict["pubmep_t1-fmi_lmi_dxa"]= databases["pubmep_t1_general"][fmi_lmi_dxap1]
    # final_dict["pubmep_t2-fmi_lmi_dxa"]= databases["pubmep_t2_general"][fmi_lmi_dxap2]
           
    # final_dict["pubmep_t1-dxa_parts"]= databases["pubmep_t1_general"][body_parts1]
    # final_dict["pubmep_t2-dxa_parts"]= databases["pubmep_t2_general"][body_parts2]
    
    
    # final_dict["pubmep_t1-fmi_lmi_tan_complete"]= pd.DataFrame.merge(databases["pubmep_t1_general"][fmi_lmi_tanp1], 
    #                                                   PGSp_t1_bioqp, on="code")
    # final_dict["pubmep_t2-fmi_lmi_tan_complete"]= pd.DataFrame.merge(databases["pubmep_t2_general"][fmi_lmi_tanp2], 
    #                                                   PGSp_t2_bioqp, on="code")
    
    # final_dict["pubmep_t1-fmi_lmi_dxa_complete"]= pd.DataFrame.merge(databases["pubmep_t1_general"][fmi_lmi_dxap1], 
    #                                                   PGSp_t1_bioqp, on="code")
    # final_dict["pubmep_t2-fmi_lmi_dxa_complete"]= pd.DataFrame.merge(databases["pubmep_t2_general"][fmi_lmi_dxap2], 
    #                                                   PGSp_t2_bioqp, on="code")
    
    # final_dict["pubmep_t1-dxa_parts_complete"]= pd.DataFrame.merge(databases["pubmep_t1_general"][body_parts1], 
    #                                                   PGSp_t1_bioqp, on="code")
    # final_dict["pubmep_t2-dxa_parts_complete"]= pd.DataFrame.merge(databases["pubmep_t2_general"][body_parts2], 
    #                                                   PGSp_t2_bioqp, on="code")
    return final_dict
    

def filtering_variables(databases):
    for db_name in databases.keys():
        for names in databases[db_name].columns:
            if "_y" in names:
                databases[db_name].drop(names, axis=1, inplace=True)
            if "_x" in names: 
                databases[db_name].rename(columns={names:names.split("_x")[0]},inplace= True)
        for names in databases[db_name].columns:
            databases[db_name].rename(columns={names:names.split("_t2")[0]},inplace= True)
            databases[db_name].rename(columns={names:names.split("_t1")[0]},inplace= True)
        if "complete" in db_name:
            databases[db_name].drop("Unnamed: 0", axis=1, inplace= True)
            databases[db_name].drop("tanner_index", axis=1, inplace= True)
                
def exporting_db(database, folder_path):
    for db_name in database.keys():
        database[db_name].to_csv(folder_path + "/" + str(db_name)+ "_processed.csv", sep= ";")
        
#MAIN

databases= open_df("C:/Users//")    
final_dict=merging(databases,["code","fmi_tan","lmi_tan"], 
                    ["code","fmi_dxa", "lmi_dxa"], 
                    ["code","tronmmagra", "tronmgrasa"],
                    ["code","brammagra", "bramgrasa"],
                    ["code","piemmagra", "piemgrasa"])

filtering_variables(final_dict)
# exporting_db(final_dict, "C:/Users//")


