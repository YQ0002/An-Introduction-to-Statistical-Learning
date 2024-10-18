rm(list = ls())
gc()

library(AmesHousing)
library(rsample)

ames <- make_ames()
dim(ames)
# ?AmesHousing::ames_raw

set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)
dim(ames_train)
dim(ames_test)

# Chapter 12 Gradient Boosting
library(dplyr)
library(gbm)
library(h2o)
library(xgboost)

h2o.init()
train_h2o <- as.h2o(ames_train)
response <- "Sale_Price"
predictors <- setdiff(colnames(ames_train), response)

set.seed(123)  # for reproducibility
ames_gbm1 <- gbm(
  formula = Sale_Price ~ .,
  data = ames_train,
  distribution = "gaussian",  # SSE loss function
  n.trees = 5000,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 10
)

# find index for number trees with minimum CV error
best <- which.min(ames_gbm1$cv.error)

# get MSE and compute RMSE
sqrt(ames_gbm1$cv.error[best])
# 22402.07


# create grid search
hyper_grid <- expand.grid(
  learning_rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  RMSE = NA,
  trees = NA,
  time = NA
)

# execute grid search
for(i in seq_len(nrow(hyper_grid))) {
  
  # fit gbm
  set.seed(123)  # for reproducibility
  train_time <- system.time({
    m <- gbm(
      formula = Sale_Price ~ .,
      data = ames_train,
      distribution = "gaussian",
      n.trees = 500, # 5000
      shrinkage = hyper_grid$learning_rate[i], 
      interaction.depth = 3, 
      n.minobsinnode = 10,
      cv.folds = 10 
    )
  })
  
  # add SSE, trees, and training time to results
  hyper_grid$RMSE[i]  <- sqrt(min(m$cv.error))
  hyper_grid$trees[i] <- which.min(m$cv.error)
  hyper_grid$Time[i]  <- train_time[["elapsed"]]
  
  print(i)
}

# results
arrange(hyper_grid, RMSE)



# search grid
hyper_grid <- expand.grid(
  n.trees = 6000,
  shrinkage = 0.01,
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15)
)

# create model fit function
model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(123)
  m <- gbm(
    formula = Sale_Price ~ .,
    data = ames_train,
    distribution = "gaussian",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    n.minobsinnode = n.minobsinnode,
    cv.folds = 10
  )
  # compute RMSE
  sqrt(min(m$cv.error))
}

# perform search grid with functional programming
hyper_grid$rmse <- purrr::pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)

# results
arrange(hyper_grid, rmse)

# n.trees shrinkage interaction.depth n.minobsinnode     rmse
# 1    6000      0.01                 5              5 21505.73
# 2    6000      0.01                 7              5 21525.52
# 3    6000      0.01                 5             10 21667.50
# 4    6000      0.01                 7             10 21706.33
# 5    6000      0.01                 3              5 21962.44
# 6    6000      0.01                 3             10 21983.03
# 7    6000      0.01                 5             15 21999.32
# 8    6000      0.01                 7             15 22189.80
# 9    6000      0.01                 3             15 22204.50




# 12.4 Stochastic GBMs

# refined hyperparameter grid
hyper_grid <- list(
  sample_rate = c(0.5, 0.75, 1),              # row subsampling
  col_sample_rate = c(0.5, 0.75, 1),          # col subsampling for each split
  col_sample_rate_per_tree = c(0.5, 0.75, 1)  # col subsampling for each tree
)

# random grid search strategy
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.001,   
  stopping_rounds = 10,         
  max_runtime_secs = 60*60      
)

# perform grid search 
grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid",
  x = predictors, 
  y = response,
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  ntrees = 6000,
  learn_rate = 0.01,
  max_depth = 7,
  min_rows = 5,
  nfolds = 10,
  stopping_rounds = 10,
  stopping_tolerance = 0,
  search_criteria = search_criteria,
  seed = 123
)

# collect the results and sort by our model performance metric of choice
grid_perf <- h2o.getGrid(
  grid_id = "gbm_grid", 
  sort_by = "mse", 
  decreasing = FALSE
)

grid_perf

# Grab the model_id for the top model, chosen by cross validation error
best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now let’s get performance metrics on the best model
h2o.performance(model = best_model, xval = TRUE)


# RMSE:  21392.57



# XGBoost
library(recipes)
xgb_prep <- recipe(Sale_Price ~ ., data = ames_train) %>%
  step_integer(all_nominal()) %>%
  prep(training = ames_train, retain = TRUE) %>%
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "Sale_Price")])
Y <- xgb_prep$Sale_Price

set.seed(123)
ames_xgb <- xgb.cv(
  data = X,
  label = Y,
  nrounds = 6000, 
  objective = "reg:squarederror",
  early_stopping_rounds = 50, 
  nfold = 10,
  params = list(
    eta = 0.1,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.8,
    colsample_bytree = 1.0),
  verbose = 0
)  

# minimum test CV RMSE
min(ames_xgb$evaluation_log$test_rmse_mean)
# 23341.22


# hyperparameter grid
hyper_grid <- expand.grid(
  eta = c(0.01, 0.05, 0.1, 0.2, 0.5, 1),
  max_depth = c(3, 5, 7, 9), 
  min_child_weight = c(3, 5, 7, 9),
  subsample = c(0.5, 0.75), 
  colsample_bytree = c(0.5, 0.75),
  gamma = 1,
  lambda = 1,
  alpha = 1,
  rmse = 0,          # a place to dump RMSE results
  trees = 0          # a place to dump required number of trees
)

# grid search
for(i in seq_len(nrow(hyper_grid))) {
  set.seed(123)
  m <- xgb.cv(
    data = X,
    label = Y,
    nrounds = 4000,
    objective = "reg:linear",
    early_stopping_rounds = 50, 
    nfold = 10,
    verbose = 0,
    params = list( 
      eta = hyper_grid$eta[i], 
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i], 
      lambda = hyper_grid$lambda[i], 
      alpha = hyper_grid$alpha[i]
    ) 
  )
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
  print(i)
}

# results
hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()


hyper_grid_sub <- subset(hyper_grid, rmse!=0)

# eta max_depth min_child_weight subsample colsample_bytree gamma lambda alpha     rmse trees
# 1  0.01         3                3       0.5              0.5     1      1     1 22376.47  3990
# 2  0.05         3                3       0.5              0.5     1      1     1 22242.83  1108
# 3  0.10         3                3       0.5              0.5     1      1     1 23148.94   576
# 4  0.20         3                3       0.5              0.5     1      1     1 25078.58   167
# 5  0.50         3                3       0.5              0.5     1      1     1 27934.54    55
# 6  1.00         3                3       0.5              0.5     1      1     1 41034.83    21
# 7  0.01         5                3       0.5              0.5     1      1     1 21827.19  2837
# 8  0.05         5                3       0.5              0.5     1      1     1 22731.64   650
# 9  0.10         5                3       0.5              0.5     1      1     1 23459.52   285
# 10 0.20         5                3       0.5              0.5     1      1     1 26129.92   142
# 11 0.50         5                3       0.5              0.5     1      1     1 30961.80    20
# 12 1.00         5                3       0.5              0.5     1      1     1 41380.36     2
# 13 0.01         7                3       0.5              0.5     1      1     1 22150.86  2429
# 14 0.05         7                3       0.5              0.5     1      1     1 22832.88   644
# 15 0.10         7                3       0.5              0.5     1      1     1 23821.06   260
# 16 0.20         7                3       0.5              0.5     1      1     1 25930.77    70
# 17 0.50         7                3       0.5              0.5     1      1     1 30915.92    14
# 18 1.00         7                3       0.5              0.5     1      1     1 40887.57     2

min(hyper_grid_sub$rmse)
# [1] 21827.19

hyper_grid_sub[7,]



# optimal parameter list
params <- list(
  eta = 0.01,
  max_depth = 5,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5
)

# train final model
xgb.fit.final <- xgboost(
  params = params,
  data = X,
  label = Y,
  nrounds = 3944,
  objective = "reg:linear",
  verbose = 0
)



# variable importance plot
vip::vip(xgb.fit.final) 
