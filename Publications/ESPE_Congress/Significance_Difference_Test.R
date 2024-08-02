library(tidyr)
library(dplyr)
library(C50)
library(pROC)

setwd("C:/Users/")
genobox_raw <- read.csv("C:/Users//", sep= ";", header = T)
g <- read.csv("C:/Users//", sep= ";", header = T)
g <- na.omit(g)
genobox_raw <- na.omit(genobox_raw)
genobox <- merge(genobox_raw, g, by= "code")
iberomics_raw <- read.csv("C:/Users//", sep= ";", header = T)
iberomics_raw <- na.omit(iberomics_raw)
i <- read.csv("C:/Users//", sep= ";", header= T)
i <- na.omit(i)
iberomics <- merge(iberomics_raw, i, by= "code")

#Processing
genobox_f= filter(genobox, genobox$sex.x==0)
genobox_m= filter(genobox, genobox$sex.x==1)
genobox_f$MetS <- factor(genobox_f$MetS)
genobox_m$MetS <- factor(genobox_m$MetS)
iberomics_f= filter(iberomics, iberomics$sex.x==0)
iberomics_m= filter(iberomics, iberomics$sex.x==1)
iberomics_f$MetS <- factor(iberomics_f$MetS)
iberomics_m$MetS <- factor(iberomics_m$MetS)

BMIz_gf= genobox_f[, c("bmi_zscore_orbegozo_longi", "MetS")]
BMIz_gm= genobox_m[, c("bmi_zscore_orbegozo_longi", "MetS")]

BMIz_if= iberomics_f[, c("bmi_zscore_orbegozo_longi", "MetS")]
BMIz_im= iberomics_m[, c("bmi_zscore_orbegozo_longi", "MetS")]

BMI_gf= genobox_f[, c("BMI", "MetS")]
BMI_gm= genobox_m[, c("BMI", "MetS")]

BMI_if= iberomics_f[, c("BMI", "MetS")]
BMI_im= iberomics_m[, c("BMI", "MetS")]

TMI_gf= genobox_f[, c("TMI", "MetS")]
TMI_gm= genobox_m[, c("TMI", "MetS")]

TMI_if= iberomics_f[, c("TMI", "MetS")]
TMI_im= iberomics_m[, c("TMI", "MetS")]


#Puntos de corte
den_MetS= density(as.numeric(BMIz_gf$bmi_zscore_orbegozo_longi[BMIz_gf$MetS == 1]))
den_noMetS= density(as.numeric(BMIz_gf$bmi_zscore_orbegozo_longi[BMIz_gf$MetS == 0]))
objroc <- roc(BMIz_gm$MetS, as.numeric(BMIz_gm$bmi_zscore_orbegozo_longi),auc=T)
plot.roc(roc2,print.auc=T,print.thres = "best",
         col="blue",xlab="1 - Specifity",ylab="Sensibility", legacy.axes = T,
         main= "ROC curve and cutoff point for BMIzscore in boys population (GenoBox)")

PC_BMIzgf= 2.372
PC_BMIzgm= 2.147

PC_BMIgf= 25.260
PC_BMIgm= 25.442
  
PC_TMIgf= 16.873
PC_TMIgm= 16.490

#pvalor
BMIz_f= ifelse(BMIz_gf$bmi_zscore_orbegozo_longi < PC_BMIzgf, 0, 1)
BMIz_m= ifelse(BMIz_gm$bmi_zscore_orbegozo_longi < PC_BMIzgm, 0, 1)

BMIf= ifelse(BMI_gf$BMI < PC_BMIgf, 0, 1)
BMIm= ifelse(BMI_gm$BMI < PC_BMIgm, 0, 1)

TMIf= ifelse(TMI_gf$TMI < PC_TMIgf, 0, 1)
TMIm= ifelse(TMI_gm$TMI < PC_TMIgm, 0, 1)

roc1 <- roc(BMI_gf$MetS, as.numeric(BMI_gf$BMI),auc=T,ci=T)
roc2 <- roc(BMIz_gf$MetS, as.numeric(BMIz_gf$bmi_zscore_orbegozo_longi),auc=T,ci=T)


roc.test(roc1, roc2, method= "delong")
