rm(list = ls())

# 8 Tree-Based Methods

# 8.1 The Basics of Decision Trees
# 8.1.1 Regression Trees
# regions: terminal nodes or leaves of the tree
# points: internal nodes

# Prediction via Stratification of the Feature Space
# 1. Divide the predictor space into j distinct and non-overlapping regions
# 2. Make prediction of the mean of the response values for the training observations in R_j

# Find boxes that minimize the RSS
# use a #top-down#, #greedy# approach known as #recursive binary splitting#.
# Min(sum((y_i - y_hat_R1)^2) + sum((y_i - y_hat_R2)^2))

# Tree Pruning
# Cost complexity pruning: indexed by a nonnegative tuning parameter α
# Use K-fold cross-validation to choose α

# Building a Regression Tree
# 1. Use recursive binary splitting to grow a large tree on the training data
# 2. Apply cost complexity pruning
# 3. Use K-fold cross-validation to choose α
# (a) Repeat Steps 1 and 2
# (b) Evaluate kth the mean squared prediction error
# 4. Return the subtree from Step 2 that corresponds to the chosen value of α


# 8.1.2 Classification Trees


# 8.1.3 Trees Versus Linear Models
# LR: f(X) = beta_0 + sum(beta_j * X_j)
# RT: f(X) = sum(c_n) * 1(X∈R_m))
# Tree behaves better when highly nonlinear and complex relationship exists


# 8.1.4 Advantages and Disadvantages of Trees
# Pro: easy to explain, human-like decision-making, graphical displayed, no dummy variables
# Con: lower predictive accuracy(can be solved), very non-robust


# 8.2 Bagging, Random Forests, Boosting, and Bayesian Additive Regression Trees

# 8.2.1 Bagging
# bootstrap
# f_avg(x) = 1/B * sum(f_hat_b (x))
# f_hat_bag(x) = 1/B * sum(f_hat_*b (x))
# record the class predicted by each of the B trees, and take a #majority vote#

# Out-of-Bag (OOB) Error Estimation
# With B sufficiently large, 
# OOB error is virtually equivalent to leave-one-out cross-validation error.

# Variable Importance Measures
# bagging improves prediction accuracy at the expense of interpretability
# using the RSS or the Gini index to interpret the importance of each predictor


# 8.2.2 Random Forests
# choose m ≈ p^(-1)
# improvement over bagged trees by a way that decorrelates the trees
# forcing each split to consider only a subset of the predictors


# 8.2.3 Boosting
# each tree is grown using information from previously grown trees. 
# Instead of CV bootstrap, 
# each tree is fit on a modified version of the original data set

# three tuning parameters
# 1. The number of trees B
# 2. The shrinkage parameter λ
# 3. The number d of splits in each tree


# 8.2.4 Bayesian Additive Regression Trees(BART)


# 8.3 Lab: Decision Trees
# 8.3.1 Fitting Classification Trees
library(tree)
library (MASS)
help(Boston)
head(Boston)

set.seed(1)
train <- sample (1:nrow(Boston), nrow(Boston) / 2)
tree.boston <- tree(medv~., Boston, subset = train)
summary(tree.boston)
nrow(Boston) / 2 - 7

plot(tree.boston)
text(tree.boston, pretty = 0)

cv.boston <- cv.tree(tree.boston)
plot(cv.boston$size , cv.boston$dev, type = "b")

prune.boston <- prune.tree(tree.boston, best = 5)
plot(prune.boston)
text(prune.boston , pretty = 0)

yhat <- predict(tree.boston, newdata = Boston[-train, ])
boston.test <- Boston[-train, "medv"]
plot(yhat, boston.test)
abline(0, 1)
mean((yhat - boston.test)^2)


# 8.3.3 Bagging and Random Forests
library(randomForest)

# bagging
set.seed(1)
bag.boston <- randomForest(medv~., data = Boston, 
                           subset = train, mtry = 13, importance = T)
bag.boston
help(Boston)
names(Boston)

yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
plot(yhat.bag, boston.test)
abline(0, 1)
mean((yhat.bag - boston.test)^2)

bag.boston <- randomForest(medv~., data = Boston, 
                           subset = train, mtry = 13, ntree = 25)
yhat.bag <- predict(bag.boston, newdata = Boston[-train, ])
mean((yhat.bag - boston.test)^2)

names(Boston)

# random forest
set.seed(1)
rf.boston <- randomForest(medv~., data = Boston, 
                          subset = train, mtry = 6, importance = T)
ncol(Boston)/3

yhat.rf <- predict(rf.boston, newdata = Boston[-train, ])
mean((yhat.rf - boston.test)^2)

importance(rf.boston) 
varImpPlot(rf.boston)


# 8.3.4 Boosting
library(gbm)
set.seed(1)

boost.boston <- gbm(medv~., data = Boston[train, ], 
                    distribution = "gaussian", n.trees = 5000,
                    interaction.depth = 4)
summary(boost.boston)

# partial dependence plot
plot(boost.boston, i = "rm")
plot(boost.boston, i = "lstat")

yhat.boost <- predict(boost.boston, 
                      newdata = Boston[-train, ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)

# The default λ value is 0.001

set.seed(1)
boost.boston <- gbm(medv~., data = Boston[train, ], 
                    distribution = "gaussian", n.trees = 5000,
                    interaction.depth = 4, shrinkage = 0.01, verbose = F)
yhat.boost <- predict (boost.boston, 
                       newdata = Boston[-train , ], n.trees = 5000)
mean((yhat.boost - boston.test)^2)



# Exercise 
# 7
library(randomForest)
library (MASS)
set.seed(1101)

dim(Boston)
head(Boston)

# Construct the train and test matrices
train = sample(dim(Boston)[1], dim(Boston)[1]/2)
X.train = Boston[train, -14]
X.test = Boston[-train, -14]
Y.train = Boston[train, 14]
Y.test = Boston[-train, 14]

p = dim(Boston)[2] - 1
p.2 = p/2
p.sq = sqrt(p)

rf.boston.p = randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test, 
                           mtry = p, ntree = 500)
rf.boston.p.2 = randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test, 
                             mtry = p.2, ntree = 500)
rf.boston.p.sq = randomForest(X.train, Y.train, xtest = X.test, ytest = Y.test, 
                              mtry = p.sq, ntree = 500)

plot(1:500, rf.boston.p$test$mse, col = "green", type = "l", xlab = "Number of Trees", 
     ylab = "Test MSE", ylim = c(10, 19))
lines(1:500, rf.boston.p.2$test$mse, col = "red", type = "l")
lines(1:500, rf.boston.p.sq$test$mse, col = "blue", type = "l")
legend("topright", c("m=p", "m=p/2", "m=sqrt(p)"), col = c("green", "red", "blue"), 
       cex = 1, lty = 1)
