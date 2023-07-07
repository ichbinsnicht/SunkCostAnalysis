setwd("C:/Users/mah1548/Desktop/SunkCostAnalysis")

mydata = read.csv("predictions.csv")

summary(lm(dependent ~ regressor, data=mydata))

summary(lm(dependent ~ regressor+predictionLinear, data=mydata))

summary(lm(dependent ~ regressor+predictionNonlinear, data=mydata))
