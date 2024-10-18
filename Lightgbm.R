# References:
# https://papers.nips.cc/paper_files/paper/2017/file/6449f44a102fde848669bdd9eb6b76fa-Paper.pdf
# https://www.datatechnotes.com/2022/04/lightgbm-regression-example-in-r.html
# https://www.kaggle.com/code/nschneider/gbm-vs-xgboost-vs-lightgbm#Results

rm(list = ls())
gc()

library(AmesHousing)
library(rsample)

library(gbm)
library(lightgbm)
library(caret)
library(ggplot2)

ames <- make_ames()
dim(ames)
names(ames)

# gbm
# ~16s

set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)
dim(ames_train)
dim(ames_test)

set.seed(123)
boost.ames <- gbm(Sale_Price~., data = ames_train, 
                    distribution = "gaussian", n.trees = 100,
                    interaction.depth = 4, 
                    cv.folds = 10)
summary(boost.ames)
names(boost.ames)

best <- which.min(boost.ames$cv.error)
sqrt(boost.ames$cv.error[best])

# RMSE: 25167.62



# Lightgbm
# ~2s

library(microbenchmark, quietly=TRUE)

# ames <- make_ames()
set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)
dim(ames_train)
dim(ames_test)

# Convert Sale_Price to numeric
train_x <- ames_train[, -79]
train_y <- as.numeric(unlist(ames_train[, 79]))
test_x <- ames_test[, -79]
test_y <- as.numeric(unlist(ames_test[, 79]))


# Convert the datasets to lgb.Dataset
lgb_train <- lgb.Dataset(data.matrix(train_x), label = train_y)
lgb_test <- lgb.Dataset(data.matrix(test_x), label = test_y)
help("lgb.Dataset")


# define parameters
params = list(objective = "regression"
              , metric = "l2"
              , min_data = 1L
              , learning_arte = 0.01
              )

# validataion data
valids <- list(test = lgb_test)

# train model
set.seed(123)
lgb_model_20 <- lgb.train(params = params, 
                   data = lgb_train, 
                   nrounds = 20L, 
                   valids = valids)
help(lgb.train)

# Get L2 values for "test" dataset
lgb.get.eval.result(lgb_model_20, "test", "l2")
sqrt(lgb.get.eval.result(lgb_model_20, "test", "l2"))
min(sqrt(lgb.get.eval.result(lgb_model_20, "test", "l2")))

help(lgb.get.eval.result)

# 20L: 27578.87

set.seed(123)
lgb_model_50 <- lgb.train(params = params, 
                       data = lgb_train, 
                       nrounds = 50L, 
                       valids = valids)
min(sqrt(lgb.get.eval.result(lgb_model_50, "test", "l2")))

# 50L: 22952.66

set.seed(123)
lgb_model_100 <- lgb.train(params = params, 
                          data = lgb_train, 
                          nrounds = 100L, 
                          valids = valids)
min(sqrt(lgb.get.eval.result(lgb_model_50, "test", "l2")))

# 100L: 22952.66



lgb.importance(lgb_model_20, percentage = T)
lgb.importance(lgb_model_50, percentage = T)


