# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 21:17:41 2023

@author: julil
"""
import pandas as pd
import numpy as np
import random
import xgboost as xgb
from sklearn.model_selection import train_test_split
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, roc_auc_score, make_scorer, roc_curve, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
import shap
import lime
import eli5
from scipy.stats import norm
import csv

#Abrimos el df y vemos que los datos están bien y podemos analizarlos

df_pasar= pd.read_csv('C:/Users/julil/Desktop/TFM/BDs_pasar/iberomics_F_base_processed.csv', sep= ';',  encoding='latin-1')
df= pd.read_csv('C:/Users/julil/Desktop/TFM/BDs_pasar/iberomics_F_base_processed.csv', sep= ";", encoding='latin-1')
df2= df.copy()
if df.isnull().any().any():
    print("Los valores nulos están en: ", df.columns[df.isna().any()].tolist())
else: 
    print("No hay valores nulos")
       
#Una vez rellenados, tendríamos que ver si los valores están balanceados o no.
# total= len(df_pasar)
# res= df_pasar.loc[:,"Insulin_resistance"].sum() #La resistencia a insulina viene marcada por 1 y la no resistencia por 0
# nores= total - res
# pres= res/ total
# pnores= nores/ total
# print("La probabilidad de ser resistente: ", pres, '\nLa probabilidad de no ser resistente: ', pnores)
Y= df.pop("homa_zscore_stavnsbo")
Y_pasar= df_pasar.pop("homa_zscore_stavnsbo")

#BÚSQUEDA DE LOS MEJORES PARÁMETROS si es que quiero hacerlo y lo meto en una función
trainX, testX, trainY, testY= train_test_split(df, Y, test_size= 0.3, random_state= 42) #El test sale con 143 individuos, de los cuales 93 son resistentes y 50 no. 
# def parametrosmejores(trainX, trainY):
#     scorers= {'accuracy': make_scorer(accuracy_score), 'f1': make_scorer(f1_score), 'auc': make_scorer(roc_auc_score), 'recall': make_scorer(recall_score)}
#     model= xgb.XGBClassifier()
#     param_grid = {
#         'max_depth': [1, 3, 5, 8, 10],
#         'learning_rate': [0.01, 0.05, 0.1, 0.5, 0.8],
#         'n_estimators': [50, 100, 200, 500]
#     }
#     skf= StratifiedKFold(n_splits=5, shuffle= True, random_state=0)
#     grid_search= GridSearchCV(model, param_grid, scoring= scorers, cv= skf, refit= 'recall')
#     grid_search.fit(trainX, trainY)
#     parametros= grid_search.best_params_
#     print("Best parameters: {}".format(parametros))
#     return parametros
# parametros= parametrosmejores(trainX, trainY)

# Ahora hacemos el 5x5 cross fold validation
def funcion_evaluacion(pruebY, y):
    m= confusion_matrix(pruebY, y.astype(int)) #La clase positiva es no tener (que es la minoritaria) y la negativa es tener. 
    precision= accuracy_score(pruebY, y.astype(int))
    f1= f1_score(pruebY, y.astype(int))
    esp= m[1,1]/(m[1,1]+m[0,1])
    sens= m[0,0]/(m[0,0]+m[1,0])
    AUC= roc_auc_score(pruebY, y.astype(int))
    fpos, tpos, matriz= roc_curve(pruebY, y.astype(int))
    plt.plot(fpos, tpos, linestyle='--')
    plt.xlabel('Falsos positivos')
    plt.ylabel('Verdaderos positivos')
    return (m, precision, f1, esp, sens, AUC)

listam, listapre, listaf1, listaesp, listasens, listaauc= [], [], [], [], [], []
listapre1, listaf11, listaesp1, listasens1, listaauc1= [], [], [], [], []
dicm, dicpre, dicf1, dicesp, dicsens, dicauc= {'media':[]}, {'media':[], 'std':[]}, {'media':[], 'std':[]}, {'media':[], 'std':[]}, {'media':[], 'std':[]}, {'media':[], 'std':[]}
modelodef= xgb.XGBClassifier() #max_depth= parametros['max_depth'], learning_rate= parametros['learning_rate'], n_estimators= parametros['n_estimators']
for aleat in range(0,5):
    rd= random.randint(0,100)
    skf= StratifiedKFold(n_splits=5, shuffle= True, random_state=rd)
    for inde, indp in skf.split(trainX, trainY):
        entrX, pruebX= trainX.iloc[inde], trainX.iloc[indp]
        entrY, pruebY= trainY.iloc[inde], trainY.iloc[indp]
        modelodef.fit(entrX, entrY)
        y=modelodef.predict(pruebX)
        plt.figure(1)
        m, precision, f1, esp, sens, AUC= funcion_evaluacion(pruebY, y)
        listam.append(m)
        listapre.append(precision)
        listaf1.append(f1)
        listaesp.append(esp)
        listasens.append(sens)
        listaauc.append(AUC)
        listapre1.append(precision)
        listaf11.append(f1)
        listaesp1.append(esp)
        listasens1.append(sens)
        listaauc1.append(AUC)
    dicm['media'].append((sum(listam)/len(listam)))
    dicpre['media'].append(np.mean(listapre)), dicpre['std'].append(np.std(listapre))
    dicf1['media'].append(np.mean(listaf1)), dicf1['std'].append(np.std(listaf1))
    dicesp['media'].append(np.mean(listaesp)), dicesp['std'].append(np.std(listaesp))
    dicsens['media'].append(np.mean(listasens)), dicsens['std'].append(np.std(listasens))
    dicauc['media'].append( np.mean(listaauc)), dicauc['std'].append(np.std(listaauc))
    listapre.clear()
    listaf1.clear()
    listaesp.clear()
    listasens.clear()
    listaauc.clear()
    listam.clear()
print("La matriz de confusión media es: {}\n".format((sum(dicm['media'])/len(dicm['media'])).astype(int)))
print("La precisión media es {:.3f} y la desviación estándar es {:.3f} ".format(np.mean(dicpre['media']), np.std(dicpre['std'])))
print("El valor de f1 medio es {:.3f} y la desviación estándar es {:.3f}".format(np.mean(dicf1['media']), np.std(dicf1['std'])))
print("El valor medio de especificidad es {:.3f} y la desviación estándar es {:.3f}".format(np.mean(dicesp['media']), np.std(dicesp['std'])))
print("El valor medio de sensibilidad es {:.3f} y la desviación estándar es {:.3f}".format(np.mean(dicesp['media']), np.std(dicesp['std'])))
print("El valor medio de AUC es {:.3f} y la desviación estándar es {:.3f}".format(np.mean(dicauc['media']), np.std(dicauc['std']))) 


#Con esto guardamos los 25 valores generados
with open('C:/Users/julil/Desktop/TFG/df/Listas1.csv', 'w', newline= '') as archivo_csv: 
    writer= csv.writer(archivo_csv)
    for l, l1, l2, l3, l4 in zip(listaf11, listapre1, listasens1, listaesp1, listaauc1):
        writer.writerow([l,l1,l2,l3,l4])


#Una vez decidimos que tenemos buen modelo, le pasamos el test real.
modelodef.fit(trainX, trainY)
y= modelodef.predict(df_pasar)
plt.figure(2)
m, precision, f1, esp, sens, auc= funcion_evaluacion(Y_pasar, y)
cm_display = ConfusionMatrixDisplay(m)
plt.figure(3)
cm_display.plot()
plt.show()
print("La matriz de confusión final es {}, la precisión {}, la f1 {}, la especificidad {}, la sensibilidad {} y la AUC {}".format(m, precision, f1, esp, sens, auc))

#Ahora pasamos a implementar los explicadores, qué ha tomado el modelo como mejor opción mirar cortes para reajustar el modelo umbral tb mirar solo exposiciones sin bioq
# SHAP
features= []
for i in df_pasar:
    features.append(i)
explainerS= shap.TreeExplainer(modelodef)
shapvalues= explainerS.shap_values(df_pasar)
# plt.figure(4)  
# # colors = ["#9bb7d4", "#0f4c81"]         
# # cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
shap.summary_plot(shapvalues,df_pasar,alpha=1, show= False)
# # plt.title('Valores SHAP calculados para el modelo')
# # plt.figure(5)
# # shap.summary_plot(shapvalues,df_pasar,plot_type='bar', alpha=1, show= False)
# # shap.summary_plot(shap_values= shapvalues, features=features, show= True, alpha= 0.8, title= 'Valores en la predicción')
# # shap.dependence_plot('ACT_percent_Tan', shapvalues, df_pasar, interaction_index= 'Alimentos_azucarados_semana')
# # shap.dependence_plot('Estadio_tanner', shapvalues, df_pasar, interaction_index= 'TSH_mU_l_')
# # shap.dependence_plot('Inicio_pescado', shapvalues, df_pasar)
# # shap.dependence_plot('TSH_mU_l_', shapvalues, df_pasar, interaction_index= 'BMI')
# # shap.dependence_plot('FSH_UI.L', shapvalues, df_pasar, interaction_index='BMI')
# # shap.dependence_plot('Height', shapvalues, df_pasar, interaction_index='Age')
# # shap.dependence_plot('Age', shapvalues, df_pasar, interaction_index='Height')
# # shap.dependence_plot('Pliegue_biceps', shapvalues, df_pasar, interaction_index='Perimetro_de_cintura')
# # shap.dependence_plot('Pliegue_suprailiaco', shapvalues, df_pasar, interaction_index='Pliegue_biceps')
# # shap.dependence_plot('Perimetro_de_cintura', shapvalues, df_pasar, interaction_index='Pliegue_suprailiaco')
# # shap.dependence_plot('Vit_D_microgramos.l', shapvalues, df_pasar, interaction_index='TSH_mU_l_')
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar, interaction_index= 'Peso_Kg')
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar, interaction_index= 'Peso_Kg')
# # plt.figure(7)
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar, interaction_index= 'Estadio_tanner')
# # plt.figure(8)
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar, interaction_index= 'BMI')

# # plt.figure(6)
# shap.dependence_plot('Alimentos_azucarados_semana', shapvalues, df_pasar, interaction_index= 'Toma_varias_veces_dia_golosinas')
# # plt.figure(7)
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar, interaction_index= 'Alimentos_azucarados_dia')
# # plt.figure(8)
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar,  interaction_index= 'Comes_Bolleria_y_con_que_frecuencia_lo_haces_mientras_juegas_video_juegos')
# # plt.show()
# # plt.figure(9)
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar,  interaction_index= 'Lacteos_sin_azucar_dia')
# # plt.figure(10)
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar, interaction_index= 'Lacteos_con_azucar_dia')
# # plt.figure(11)
# # shap.dependence_plot('Cortisol__microgramos_dl_', shapvalues, df_pasar, interaction_index= 'Alimentos_azucarados_semana')
# shap.dependence_plot('Salsas_semana', shapvalues, df_pasar, interaction_index= 'Salsas_semana')
# shap.dependence_plot('Salsas_dia', shapvalues, df_pasar, interaction_index= 'Salsas_dia')
# shap.dependence_plot('Alimentos_azucarados_semana', shapvalues, df_pasar, interaction_index= 'Alimentos_azucarados_semana')
# shap.dependence_plot('Toma_pasta_o_arroz_casi_a_diario___5_semana_', shapvalues, df_pasar, interaction_index= 'Toma_pasta_o_arroz_casi_a_diario___5_semana_')
# shap.dependence_plot('Inicio_legumbres', shapvalues, df_pasar, interaction_index= 'Inicio_legumbres')
# # shap.dependence_plot('Inicio_pescado', shapvalues, df_pasar, interaction_index= 'Inicio_pescado')
# shap.dependence_plot('FastFood_semana', shapvalues, df_pasar, interaction_index= 'FastFood_semana')
# shap.dependence_plot('Inicio_huevo', shapvalues, df_pasar, interaction_index= 'Inicio_huevo')
# shap.dependence_plot('Frutas_Verduras_dia', shapvalues, df_pasar, interaction_index= 'Frutas_Verduras_dia')
# shap.dependence_plot('TV_mañana', shapvalues, df_pasar, interaction_index= 'TV_mañana')
# shap.dependence_plot('Con.que.frecuencia.tomas.Queso.curado.semi.al.dia.', shapvalues, df_pasar, interaction_index= 'Con.que.frecuencia.tomas.Queso.curado.semi.al.dia.')
# shap.dependence_plot('Inicio_verduras', shapvalues, df_pasar, interaction_index= 'Inicio_verduras')
# #GRÁFICOS DE FUERZA
# # NIÑOS PREPÚBERES
# shap.force_plot(explainerS.expected_value, shapvalues[7], df_pasar.iloc[[7]], matplotlib= True) #Niño pequeño no IR
plt.figure()
shap.plots.waterfall(explainerS(df_pasar)[7], max_display= 20)
# shap.force_plot(explainerS.expected_value, shapvalues[47], df_pasar.iloc[[47]], matplotlib= True) #Niño pequeño no IR
# shap.plots.waterfall(explainerS(df_pasar)[48], max_display= 40)




# explainerL= lime.lime_tabular.LimeTabularExplainer(df_pasar.values, feature_names= features, kernel_width=3)
# listaLime= [9]  


# for i in listaLime:
#     exp= explainerL.explain_instance(df_pasar.iloc[[i]].values[0], modelodef.predict_proba, num_features=40)
#     archivo= f"Explicador_{i}.html"
#     exp.save_to_file(archivo)
    
# # # # #ELI5 hay que hacerlo en un Notebook
# eli5.show_prediction(modelodef, df_pasar.iloc[[7]] , feature_names= features, show_feature_values= True, top= 30)
# eli5.show_prediction(modelodef, df_pasar.iloc[[47]] , feature_names= features, show_feature_values= True, top= 30)
