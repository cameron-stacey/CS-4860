iris <- read.csv("~/CS 4860/Homework 1/iris.csv")
flowercode <- ifelse(iris$flowerclass == "Iris-setosa", 1, 0)
model <- glm(flowercode~iris$sepallength + iris$sepalwidth + iris$petallength + iris$petalwidth)
summary(model)