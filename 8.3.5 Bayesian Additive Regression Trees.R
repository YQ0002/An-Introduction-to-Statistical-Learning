rm(list = ls())

# file:///D:/a1%E6%96%87%E4%BB%B6/2023%E5%AF%8C%E5%B1%B1/ISLRv2_corrected_June_2023.pdf
# P-359

# 8.3.5 Bayesian Additive Regression Trees
library (MASS)
library(BART)
names(Boston)

set.seed(123)
train <- sample (1:nrow(Boston), nrow(Boston) / 2)

x <- Boston[, 1:12]
y <- Boston[, "medv"]
xtrain <- x[train, ]
ytrain <- y[train]
xtest <- x[-train, ]
ytest <- y[-train]

set.seed(123)
bartfit <- gbart(xtrain, ytrain, x.test = xtest)
yhat.bart <- bartfit$yhat.test.mean
mean((ytest - yhat.bart)^2)
# 14.29152

ord <- order(bartfit$varcount.mean, decreasing = T)
bartfit$varcount.mean[ord]


# Exercise 
# 8.
# Lab
library(tree)
library(ISLR2)
attach(Carseats)
names(Carseats)
# (a) Split the data set into a training set and a test set.
set.seed(123)
n <- nrow(Carseats)
index <- sample(1:n, n/2)
train <- Carseats[index,]
test <- Carseats[-index,]

# (b) Fit a regression tree to the training set.
set.seed(123)
rt.train <- tree(Sales~.- Sales, Carseats)
plot(rt.train)
text(rt.train, pretty = 0)
# What test MSE
y.te <- t(test["Sales"])
yhat.te <- predict(rt.train, test)
mean((yhat.te - y.te)^2)
# 2.619999

# (c) Use cross-validation to determine the optimal level of tree complexity.
set.seed(123)
cv.train <- cv.tree(rt.train)
names(cv.train)
cv.train
best <- cv.train$size[which.min(cv.train$dev)]
best

par(mfrow=c(1,2))
plot(cv.train$size, cv.train$dev, type = "b")
# plot(cv.train$k, cv.train$dev, type = "b")

prune.carseats <- prune.tree(rt.train, best = best)
plot(prune.carseats)
text(prune.carseats, pretty = 0)

# Does pruning the tree improve the test MSE
yhat.te <- predict(prune.carseats, test)
mean((yhat.te - y.te)^2)
# 4.466452

# (d) Use the bagging approach in order to analyze this data
library(randomForest)
set.seed(123)
bag <- randomForest(Sales~., data = Carseats, subset = index)
help("randomForest")
yhat.bag <- predict(bag, test)
mean((yhat.bag - y.te)^2)
# 3.565421
importance(bag) 

# (e) Use random forests to analyze this data.
set.seed(123)
rf <- randomForest(Sales~., data = Carseats, subset = index,
                   mtry = 5, importance=T)
yhat.rf <- predict(rf, newdata=test)
mean((yhat.rf - y.te)^2)
# 3.01707
importance(rf) 

# (f) Now analyze the data using BART.
library(BART)
dim(train)
xtrain <- train[, 2:11]
ytrain <- train[, 1]
xtest <- test[, 2:11]
ytest <- test[, 1]
set.seed(123)
bart <- gbart(xtrain, ytrain, x.test=xtest)
yhat.bart <- bart$yhat.test.mean
mean((yhat.bart - y.te)^2)
# 1.622453







