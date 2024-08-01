#User: Julia

setwd("/Users/julil/Desktop/Pruebas/")

#Loading RandomForest and Iris Import
library("randomForest")
library("dplyr")
library("caTools")
library("caret")
library("r2pmml")
library("reticulate")
#write.csv(iris, "iris.csv")

#Importing CSV
iris= read.csv("iris.csv", sep=";")
iris= subset(iris, select= -X)

iris$Species= as.factor(iris$Species)


#Splitting data into training and test
split= caTools::sample.split(iris[,1], SplitRatio = 0.7)
iris_train= subset(iris, split== "TRUE")
iris_test= subset(iris, split== "FALSE")

target= as.factor(subset(iris_test, select= Species)[,1])
target=as.integer(target)
iris_test= subset(iris_test, select= -Species)

#Parameters
randomForest::tuneRF(iris_train, as.factor(iris_train$Species), stepFactor = 1, 
                     improve= 0.1, trace= TRUE, plot= TRUE)

#MODEL
model= randomForest::randomForest(formula=Species~., data=iris_train)

#Testing model
prediction= predict(model, iris_test)
caret::confusionMatrix(as.factor(prediction), as.factor(target))


#PMML
r2pmml::r2pmml(model, file= "model.pmml")


