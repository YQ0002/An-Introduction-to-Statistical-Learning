rm(list = ls())
# 5 Resampling Methods

# 5.1 Cross-validation
# test error & training error

# 5.1.1 The Validation Set Approach
# library(ISLR2)
# help(Auto)
# mpg1 <- lm(mpg~horsepower, data = Auto)
# mpg2 <- lm(mpg~horsepower+I(horsepower^2), data = Auto)
# library(stargazer)
# stargazer(mpg1, mpg2, type = "text")
# names(summary(mpg1))
# summary(mpg1)$sigma^2
# summary(mpg2)$sigma^2

# Pros: simple and easy to apply
# Cons: test error can be highly variable; 
#       only represent a part of the data

# 5.1.2 Leave-One-Out Cross-Validation
# (LOOCV)
# CV = (1/n)*sum(MSE_i)
# CV = (1/n)*sum(((y_i-y_i_hat)/(1-h_i))^2)

# 5.1.3 k-Fold Cross-Validation
# CV = (1/k)*sum(MSE_i)

# 5.1.4 Bias-Variance Trade-Off for k-Fold Cross-Validation

# 5.1.5 Cross-Validation on Classification Problems
# CV = (1/k)*sum(Err_i)
# Err_i = I, (y_i≠y_i_hat)

# 5.2 The Bootstrap
#  minimize Var(αX + (1-α)Y)
# α = (σ_y^2 - σ_xy) / (σ_x^2 + σ_y^2 - 2*σ_xy) 

# Z: bootstrap data set
# B: times of sampling
# SE_B(α_hat) = (1/(B-1) * sum((α_i_hat - 1/B*sum(α_i_hat'))^2) )^-1

# 5.3 Lab: Cross-Validation and the Bootstrap

# 5.3.1 The Validation Set Approach

library(ISLR2)
help(Auto)

set.seed(1)
train <- sample(392, 196)
dim(Auto)
length(train)

lm.fit <- lm(mpg~horsepower, data = Auto, subset = train)
summary(lm.fit)

attach(Auto)
mean((mpg - predict(lm.fit, Auto))[-train]^2)
# MSE for the linear regression fit is 23.26601

lm.fit2 <- lm(mpg~poly(horsepower,2), data = Auto, subset = train)
summary(lm.fit2)
mean((mpg - predict(lm.fit2, Auto))[-train]^2)
# 18.71646
lm.fit3 <- lm(mpg~poly(horsepower,3), data = Auto, subset = train)
summary(lm.fit3)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)
# 18.79401

help(poly)
# poly(horsepower,3,raw = FALSE)
# Orthogonal polynonials
help(set.seed)
set.seed(2)
train <- sample(392, 196)
lm.fit <- lm(mpg~horsepower, data = Auto, subset = train)
mean((mpg - predict(lm.fit, Auto))[-train]^2)
# 25.72651
lm.fit2 <- lm(mpg~poly(horsepower,2), data = Auto, subset = train)
mean((mpg - predict(lm.fit2, Auto))[-train]^2)
# 20.43036
lm.fit3 <- lm(mpg~poly(horsepower,3), data = Auto, subset = train)
mean((mpg - predict(lm.fit3, Auto))[-train]^2)
# 20.38533

# 5.3.2 Leave-One-Out Cross-Validation
glm.fit <- glm(mpg~horsepower, data = Auto)
coef(glm.fit)

lm.fit <- lm(mpg~horsepower, data = Auto)
coef(lm.fit)

library(boot)
glm.fit <- glm(mpg~horsepower, data = Auto)
cv.err <- cv.glm(Auto, glm.fit)
cv.err$delta
help(cv.glm)

cv.error = rep(0,5)
for (i in 1:5) {
  glm.fit <- glm(mpg~poly(horsepower, i), data = Auto)
  cv.error[i] <- cv.glm(Auto, glm.fit)$delta[1]
}
cv.error
plot(cv.error)
lines(cv.error)

# 5.3.3 k-Fold Cross-Validation
set.seed(17)
cv.error.10 <- rep(0,10)
for (i in 1:10) {
  glm.fit <- glm(mpg~poly(horsepower, i), data = Auto)
  cv.error.10[i] <- cv.glm(Auto, glm.fit, K=10)$delta[1]
}
cv.error.10
plot(cv.error.10)
lines(cv.error.10)

# 5.3.4 The Bootstrap
help(Portfolio)
head(Portfolio)
alpha.fn <- function(data,index){
  X <- data$X[index]
  Y <- data$Y[index]
  return((var(Y)-cov(X,Y))/(var(X)+var(Y)-2*cov(X,Y)))
}

alpha.fn(Portfolio, 1:100)
# 0.5758321
set.seed(7)
alpha.fn(Portfolio, sample(100,100,replace = T))
# 0.5385326

boot(Portfolio, alpha.fn, R=1000)
help(boot)
names(boot(Portfolio, alpha.fn, R=1000))
boot(Portfolio, alpha.fn, R=1000)$t0
# α_hat = 0.5758321

# Estimating the Accuracy of a Linear Regression Model
boot.fn <- function(data,index)
  return(coef(lm(mpg~horsepower, data = data, subset = index)))
boot.fn(Auto, 1:392)

set.seed(1)
boot.fn(Auto, sample(392, 392, replace = T))
help(sample)
boot(Auto, boot.fn, 1000)

summary(lm(mpg~horsepower, data = Auto))

boot.fn <- function(data, index)
  coef(lm(mpg~horsepower+I(horsepower^2), data = data, subset = index))
set.seed(1)
boot(Auto, boot.fn, 1000)

# 5.4 Exercises
# Conceptual

# 1. 
#   Var(αX + (1-α)Y)
# = α^2 * Var(X) + 2α(1-α) * Cov(X,Y) + (1-α)^2 * Var(Y) 
# = α^2 * σ_X^2 + 2α(1-α) * σ_XY + (1-α)^2 * σ_Y^2 

# α is minimized when dy/dα = 0, which is:  
# 2α*σ_X^2 + (2*(1-α)-2α)*σ_XY + (-2(1-α))*σ_y^2 = 0
# 2α*σ_X^2 + (2-4α)*σ_XY + (2α-2)*σ_y^2 = 0
# 2α*σ_X^2 - 4α*σ_XY + 2α*σ_y^2 = 2*σ_y^2 - 2*σ_XY 
# α*(σ_X^2 + 2*σ_XY + σ_y^2) = σ_y^2 - σ_XY
# α = (σ_y^2 - σ_XY) / (σ_X^2 + σ_y^2 - 2*σ_XY)

# 2.
# (a) What is the probability that the first bootstrap observation 
#     is not the jth observation from the original sample? 
# 1 - 1/n

# (b) What is the probability that the second bootstrap observation 
#     is not the jth observation from the original sample?
# 1 - 1/n

# (c) Argue that the probability that the jth observation 
#     is not in the bootstrap sample is (1 − 1/n)^n.

# (d) When n = 5, what is the probability that the jth observation 
#     is in the bootstrap sample?
1 - (1 - 1/5)^5

# (e) When n = 100
n <- 100
1 - (1 - 1/n)^n

# (f) When n = 10000
n <- 10000
1 - (1 - 1/n)^n

# (g) When n = 1 to 10000
boot.p <- function(n) return(1 - (1 - 1/n)^n)
x <- 1:10000
boot.p(x)
plot(boot.p(x))
lines(boot.p(x))

# (h) When n = 100 and j = 4
store <- rep(NA, 10000)
for(i in 1:10000){
  store[i] <- sum(sample(1:100 , rep=TRUE) == 4) > 0
}
mean(store)

# 3. Review k-fold cross-validation
# (b)
# Pros: simple and easy to apply
# Cons: test error can be highly variable; 
#       only represent a part of the data
# LOOCV: k = n
 
# Applied
# 5. 
# (a)
library(ISLR)
help(Default)
attach(Default)
set.seed(1)
glm.fit = glm(default~income+balance, data = Default, family = binomial)
summary(glm.fit)

# (b)
def <- function() {
  # i.
  train <- sample(dim(Default)[1], dim(Default)[1]/2)
  # ii.
  glm.fit <- glm(default ~ income + balance, data = Default, family = binomial, 
                subset = train)
  # iii.
  glm.pred <- rep("No", dim(Default)[1]/2)
  glm.probs <- predict(glm.fit, Default[-train, ], type = "response")
  glm.pred[glm.probs > 0.5] = "Yes"
  # iv.
  return(mean(glm.pred != Default[-train, ]$default))
}
def()

# (c)
c(def(), def(), def())

# (d)
train <- sample(dim(Default)[1], dim(Default)[1]/2)
glm.fit <- glm(default ~ income + balance + student, data = Default, family = binomial, 
              subset = train)
glm.pred <- rep("No", dim(Default)[1]/2)
glm.probs <- predict(glm.fit, Default[-train, ], type = "response")
glm.pred[glm.probs > 0.5] = "Yes"
mean(glm.pred != Default[-train, ]$default)

# 6. 
# (a)
attach(Default)
set.seed(1)
glm.fit <- glm(default ~ income + balance, data = Default, family = binomial)
summary(glm.fit)

# (b)
boot.fn <- function(data, index) 
  return(coef(glm(default ~ income + balance, data = data, family = binomial, subset = index)))

# (c)
library(boot)
boot(Default, boot.fn, 50)

# 7. 
help(Weekly)
attach(Weekly)
set.seed(1)

# (a)
glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly, family = binomial)
summary(glm.fit)

# (b)
glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-1, ], family = binomial)
summary(glm.fit)

# (c)
predict.glm(glm.fit, Weekly[1, ], type = "response") > 0.5

# (d)
count <- rep(0, dim(Weekly)[1])
for (i in 1:(dim(Weekly)[1])) {
  glm.fit <- glm(Direction ~ Lag1 + Lag2, data = Weekly[-i, ], family = binomial)
  is_up <- predict.glm(glm.fit, Weekly[i, ], type = "response") > 0.5
  is_true_up <- Weekly[i, ]$Direction == "Up"
  if (is_up != is_true_up) 
    count[i] = 1
}
sum(count)

# (e)
mean(count)

# 8. 
# (a)
set.seed(1)
x <- rnorm(100)
y <- x - 2 * x^2 + rnorm(100)
# n=100, p=2

# (b)
plot(x,y)

# (c)
library(boot)
data <- data.frame(x, y)
set.seed(3)
cv.error = rep(0,4)
for (i in 1:4) {
  glm.fit <- glm(y ~ poly(x, i), data = data)
  cv.error[i] <- cv.glm(data, glm.fit)$delta[1]
}
cv.error
plot(cv.error)
lines(cv.error)

# (d)
set.seed(7)
cv.error = rep(0,4)
for (i in 1:4) {
  glm.fit <- glm(y ~ poly(x, i), data = data)
  cv.error[i] <- cv.glm(data, glm.fit)$delta[1]
}
cv.error
plot(cv.error)
lines(cv.error)

# 9. 
library(MASS)
help(Boston)
set.seed(1)
attach(Boston)

# (a)
medv.mean <- mean(medv)
medv.mean

# (b)
medv.err <- sd(medv) / sqrt(length(medv))
medv.err

# (c)
library(boot)
boot.fn <- function(data, index)
  return(mean(data[index]))
boot(medv, boot.fn, 1000)

# (d)
t.test(medv)
medv.boot <- boot(medv, boot.fn, 1000)
names(medv.boot)
c(medv.boot$t0 - 2*0.4106622, medv.boot$t0 + 2*0.4106622)

# (e)
medv.med <- median(medv)
medv.med

# (f)
boot.fn <- function(data, index) 
  return(median(data[index]))
boot(medv, boot.fn, 1000)
boot(medv, boot.fn, 1000)$t0

# (g)
medv.tenth <- quantile(medv, 0.1)
medv.tenth

# (h)
boot.fn <- function(data, index) 
  return(quantile(data[index], 0.1))
boot(medv, boot.fn, 1000)
boot(medv, boot.fn, 1000)$t0




