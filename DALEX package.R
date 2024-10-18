
library(ranger)
library(gbm)
library(DALEX)

data(titanic_imputed, package = "DALEX")
dim(titanic_imputed)

# Fit a random forest
ranger_model <- ranger(survived~., data = titanic_imputed, classification = TRUE, probability = TRUE)
gbm_model <- gbm(survived~., data = titanic_imputed, distribution = "bernoulli")

explainer_ranger <- DALEX::explain(ranger_model, 
                            data = titanic_imputed, 
                            y = titanic_imputed$survived, 
                            label = "Ranger Model")
explainer_gbm <- DALEX::explain(gbm_model, 
                                   data = titanic_imputed, 
                                   y = titanic_imputed$survived, 
                                   label = "Boosting")

fi_ranger <- model_parts(explainer_ranger, 
                         type = "ratio",
                         N = NULL, 
                         B = 10)
plot(fi_ranger)
help(model_parts)

fi_gbm <- model_parts(explainer_gbm, 
                         type = "ratio",
                         N = NULL, 
                         B = 10)
plot(fi_gbm)

plot(fi_ranger, fi_gbm)


# ALE Plot - One model
ale_ranger <- model_profile(explainer_ranger, 
                            variables = "fare", 
                            type = "accumulated") #ALE
plot(ale_ranger)

# PDP Plot - One model 

# Gender
pdp_ranger_gender <- model_profile(explainer_ranger, 
                            variables = "gender", 
                            type = "partial",
                            variable_type='categorical') #PDP
pdp_gbm_gender <- model_profile(explainer_gbm, 
                                   variables = "gender", 
                                   type = "partial",
                                   variable_type='categorical') #PDP
plot(pdp_ranger_gender, pdp_gbm_gender)


# Age
pdp_ranger_age <- model_profile(explainer_ranger, 
                                   variables = "age", 
                                   type = "partial", 
                                   N = NULL, 
                                   B = 5) #PDP
pdp_gbm_age <- model_profile(explainer_gbm, 
                                variables = "age", 
                                type = "partial") #PDP
plot(pdp_ranger_age, pdp_gbm_age)


help(model_profile)


#############################################
# load required packages
library(rsample)
library(dplyr)
library(h2o)
library(DALEX)

# initialize h2o session
h2o.no_progress()
h2o.init()

# classification data
df <- rsample::attrition %>% 
  mutate_if(is.ordered, factor, ordered = FALSE) %>%
  mutate(Attrition = recode(Attrition, "Yes" = "1", "No" = "0") %>% factor(levels = c("1", "0")))

# convert to h2o object
df.h2o <- as.h2o(df)

# create train, validation, and test splits
set.seed(123)
splits <- h2o.splitFrame(df.h2o, ratios = c(.7, .15), destination_frames = c("train","valid","test"))
names(splits) <- c("train","valid","test")

# variable names for resonse & features
y <- "Attrition"
x <- setdiff(names(df), y) 

pred <- function(model, newdata){
  results <- as.data.frame(h2o.predict(model, as.h2o(newdata)))
  return(results[[3L]])
  }





