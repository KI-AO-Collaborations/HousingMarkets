library("readxl")
library("scatterplot3d")
library(ggplot2)
data <- read_excel("Cleaned Data (1).xlsx")

#scatterplot3d(data$longitude, data$latitude, data$price, pch=20)

qplot(data$latitude, data$longitude, data=data, colour = data$regionname)

X <- data[,c(8,9,11)]
y <- data[,16]

y_col <- rainbow(length(unique(data$regionname)))

pairs(X, lower.panel = NULL, col=y_col[y])


qplot(log(data$price), log(data$landsize), colour=data$regionname)
qplot(log(data$price), log(data$buildingarea), colour=data$regionname)

a<- ggplot(data, aes(data$regionname)) +
  geom_density(data$price, color=data$regionname, inherit.aes = TRUE)

hist(data$price, breaks=500, freq = FALSE)
