# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 20:20:29 2024

@author: julil
"""

from pypmml import Model
import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from sklearn_pmml_model.ensemble import PMMLForestClassifier
import csv

#setosa=1, versicolor=2, virginica=3

model2=PMMLForestClassifier(pmml="C:/Users/julil/Desktop/Pruebas/model.pmml")
#model = Model.fromFile('C:/Users/julil/Desktop/Pruebas/model.pmml')

iris=pd.read_csv("C:/Users/julil/Desktop/Pruebas/iris.csv", sep=",")
iris= iris.drop(iris.columns[0], axis=1)
#iris.Species.replace(["setosa","versicolor","virginica"], [1,2,3], inplace=True)

target=iris.pop("Species")


result= model2.predict(iris)
explainerS= shap.Explainer(model2.predict, iris)
shap_values=explainerS(iris)

values= shap_values.values
# features= []
# for col in iris.columns:
#     features.append(col)
    
# output= "shap_values_Py.csv"
# with open(output, mode='w', newline='') as file:
#     writer = csv.writer(file)
    
#     # Escribe el encabezado con los nombres de las características
#     writer.writerow(i for i in features)
#     # Escribe los valores SHAP para cada instancia
#     for shap_row in values:
#         row = list(shap_row)
#         writer.writerow(row)

shap.plots.waterfall(explainerS(iris)[149], max_display= 4)
# shap.summary_plot(shap_values.values,iris,alpha=1, show= False)