library(tidyr)
library(dplyr)
library(C50)
library(pROC)

setwd("C:/Users//")
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

BMIz_if= iberomics_f[, c("TMI", "MetS")]
BMIz_im= iberomics_m[, c("TMI", "MetS")]

#Puntos de corte
den_MetS= density(as.numeric(BMIz_gf$BMI[BMIz_gm$MetS == 1]))
den_noMetS= density(as.numeric(BMIz_gf$bmi_zscore_orbegozo_longi[BMIz_gf$MetS == 0]))
objroc <- roc(BMIz_gm$MetS, as.numeric(BMIz_gm$bmi_zscore_orbegozo_longi),auc=T,ci=T)
plot.roc(objroc,print.auc=T,print.thres = "best",
         col="blue",xlab="1 - Specifity",ylab="Sensibility", legacy.axes = T,
         main= "ROC curve and cutoff point for BMIzscore in boys population (GenoBox)")
# PC2= 19.705
PC= 16.87
plot(NULL,xlim=c(-4, 10),ylim= c(0, 0.5), type="n",
     xlab="BMIZscore",ylab="Density", main= "Density Curve for BMIzscore in boys population (genobox)")

lines(den_MetS,col="red")
lines(den_noMetS,col="blue")
text(1.7,0.4, "Healthy")
text(3,0.45, "Unhealthy")
polygon(den_noMetS, col = rgb(0, 0, 1, alpha = 0.5), border = NA)
polygon(den_MetS, col = rgb(1, 0, 0, alpha = 0.5), border = NA)
abline(v=PC,col="green") 


#Validation
XMI_im_co= ifelse(BMIz_if$TMI < PC, 0, 1)

objroc1 <- roc(BMIz_if$MetS,XMI_im_co,auc=T,ci=T)
plot.roc(objroc1,print.auc=T,
         col="blue",xlab="1 -Specifity",ylab="Sensibility", legacy.axes = T,
         main= "ROC curve for BMIzscore in boys (iberomics validation)", print.thres = "best")

den_ifMetS= density(as.numeric(BMIz_if$bmi_zscore_orbegozo_longi[BMIz_if$MetS == 1]))
den_ifnoMetS= density(as.numeric(BMIz_if$bmi_zscore_orbegozo_longi[BMIz_if$MetS == 0]))

plot(NULL,xlim=c(-4,10),ylim= c(0, 0.5), type="n",
     xlab="BMIzScore",ylab="Density", main= "Density Curve for BMIzscore in girls population (iberomics)")

lines(den_ifMetS,col="red")
lines(den_ifnoMetS,col="blue")
text(2,0.48, "Healthy")
text(3.7,0.39, "Unhealthy")
polygon(den_ifnoMetS, col = rgb(0, 0, 1, alpha = 0.5), border = NA)
polygon(den_ifMetS, col = rgb(1, 0, 0, alpha = 0.5), border = NA)
abline(v=PC,col="green") 

