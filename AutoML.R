rm(list = ls())
gc()

library(AmesHousing)
library(rsample)

ames <- make_ames()
dim(ames)
summary(ames)
# ?AmesHousing::ames_raw

set.seed(123)
split <- initial_split(ames, prop = 0.7, strata = "Sale_Price")
ames_train <- training(split)
ames_test <- testing(split)
dim(ames_train)
dim(ames_test)

library(h2o)

h2o.init()
train_h2o <- as.h2o(ames_train)
test_h2o <- as.h2o(ames_test)
response <- "Sale_Price"
predictors <- setdiff(colnames(ames_train), response)

# H2o.autoML

aml <- h2o.automl(x = predictors, y = response,
                  training_frame = train_h2o,
                  leaderboard_frame = test_h2o, 
                  # max_models = 20,
                  max_runtime_secs = 60*5, 
                  nfolds = 10, 
                  seed = 123, 
                  sort_metric = "RMSE")
help("h2o.automl")

# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))
# AutoML, 1min, GBM_2_AutoML_2_20230726_110836 21232.90


M1 <- h2o.getModel(lb[5,"model_id"])
M1

M2 <- h2o.getModel(lb[39,"model_id"])
M2

# Plot Variable Importances
h2o.varimp_plot(M1, num_of_features = 15)
help(h2o.varimp_plot)

help("h2o.partialPlot")
h2o.partialPlot(M1, 
                train_h2o,
                cols = "Year_Built", 
                plot_stddev = F)


# automl: try 30, 60, 120 mins, importance plots, PDP


aml_30 <- h2o.automl(x = predictors, y = response,
                  training_frame = train_h2o,
                  leaderboard_frame = test_h2o, 
                  # max_models = 20,
                  max_runtime_secs = 60*30, 
                  nfolds = 10, 
                  seed = 123, 
                  sort_metric = "RMSE")

lb_30 <- aml_30@leaderboard
print(lb_30, n = nrow(lb_30))
M1_30 <- h2o.getModel(lb_30[4,"model_id"])
M1_30
# 20282.13

h2o.varimp_plot(M1_30, num_of_features = 15)
h2o.partialPlot(M1_30, 
                train_h2o,
                cols = "Year_Built", 
                plot_stddev = F)



aml_60 <- h2o.automl(x = predictors, y = response,
                     training_frame = train_h2o,
                     leaderboard_frame = test_h2o, 
                     # max_models = 20,
                     max_runtime_secs = 60*60, 
                     nfolds = 10, 
                     seed = 123, 
                     sort_metric = "RMSE")

lb_60 <- aml_60@leaderboard
print(lb_60, n = nrow(lb_60))

M1_60 <- h2o.getModel(lb_60[1,"model_id"])
M1_60
# 20282.13 


aml_120 <- h2o.automl(x = predictors, y = response,
                     training_frame = train_h2o,
                     leaderboard_frame = test_h2o, 
                     # max_models = 20,
                     max_runtime_secs = 60*120, 
                     nfolds = 10, 
                     seed = 123, 
                     sort_metric = "RMSE")

lb_120 <- aml_120@leaderboard
print(lb_120, n = nrow(lb_120))

M1_120 <- h2o.getModel(lb_120[3,"model_id"])
M1_120
# 20559.58    



# 8 Global Model-Agnostic Methods

# 8.1 Partial Dependence Plot (PDP)

# 1) Select feature. 
# 2) Define grid. 
# 3) Per grid value: 
    # a) Replace feature with grid value and 
    # b) average predictions. 
# 4) Draw curve

# Marginal effect of 1~2 features have on the predicted outcome of a ML model
# Features in C are not correlated with the features in S

# Pro: intuitive; 
#      interpretation is clear; 
#      easy to implement; 
#      causal interpretation

# Con: two features most; 
#      do not show the feature distribution; 
#      assumption of independence; 
#      Heterogeneous effects might be hidden

library(DALEX)
# https://rpubs.com/zoujinhong/1013577
# http://xai-tools.drwhy.ai/DALEX.html
help(explain)
pred <- function(M1_30, train_h2o){
  results <- as.data.frame(h2o.predict(M1_30, as.h2o(train_h2o)))
  return(results[[3L]])
}

exp_M1_30 <- DALEX::explain(M1_30, 
                     data = train_h2o, 
                     y = as.vector(train_h2o$Sale_Price))

# exp_M1_30_sort <- h2o.arrange(exp_M1_30$data, "Sale_Price")
# head(exp_M1_30$data)

fi_M1_30 <- model_parts(exp_M1_30, type = "variable_importance")
plot(fi_M1_30)



help(model_parts)

per_M1_30 <- model_performance(exp_M1_30)
plot(per_M1_30)








h2o.varimp_plot(M1_30, num_of_features = 15)
h2o.partialPlot(M1_30, 
                train_h2o,
                cols = "Year_Built", 
                plot_stddev = F)

h2o.varimp_plot(M1_60, num_of_features = 15)
h2o.partialPlot(M1_60, 
                train_h2o,
                cols = "Year_Built", 
                plot_stddev = F)

h2o.varimp_plot(M1_120, num_of_features = 15)
h2o.partialPlot(M1_120, 
                train_h2o,
                cols = "Year_Built", 
                plot_stddev = F)

# 8.2 Accumulated Local Effects (ALE) Plot
# how features influence the prediction of a ML model on average
# faster and unbiased alternative to PDP

# ALE calculates differences in predictions instead of averages
# It can show interaction effect of two features

# Pro: unbiased; 
#      faster to compute; 
#      interpretation is clear; 
#      can be decomposed

# Con: intervals is not permissible; 
#      not accompanied by ICE curves
#      unstable Second-order ALE estimates

# Usually use ALE instead of PDP







# 8.3 Feature Interaction

# 8.3.2 Theory: Friedman’s H-statistic

# estimate the interaction strength of the features
# measure how much of the variation of the prediction depends on the interaction

# Calculate the variance of the output of the PD or of the entire function

# Pro: has an underlying theory; 
#      has a meaningful interpretation; 
#      the statistic is dimensionless; 
#      detects all kinds of interactions;
#      possible to analyze higher interactions(3 or more features)

# Con: computationally expensive; 
#      estimates also have a certain variance, and unstable;
#      unstable Second-order ALE estimates



# 8.4 Functional Decomposition

# deconstructs the high-dimensional function and expresses it as a sum of 
# individual feature effects and interaction effects that can be visualized

# the decomposition is based on accumulated local effect plots

# Pro: theoretical justification; 
#      provides a better understanding of other methods; 
#      the use of ALE plots offers many advantages 

# Con: limits for high-dimensional components; 
#      individual disadvantages;
#      more appropriate for analyzing tabular data than text or images







# 8.5 Permutation Feature Importance

# measures the increase in the prediction error of the model 
# after we permuted the feature’s values

# Pro: Nice interpretation; 
#      provides a highly compressed, global insight; 
#      comparable across different problems;
#      takes into account all interactions with other features;
#      does not require retraining the model

# Con: linked to the error of the model; 
#      need access to the true outcome;
#      When the permutation is repeated, the results might vary greatly;
#      can be biased by unrealistic data instances;
#      Adding a correlated feature can decrease the importance of the associated feature;



