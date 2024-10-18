rm(list = ls())
# 6. Linear Model Selection and Regularization

# 6.1 Subset Selection
# C_p, (AIC), BIC, or adjusted R2


# 6.2 Shrinkage Methods
# shrinking the coefficient estimates can reduce the variance

# 6.2.1 Ridge Regression
# OLS: 
# RSS = sum(y_i - beta_0 - sum(beta_j * x_ij))^2
# Ridge Regression: (shrinkage penalty)
# beta_R_hat = RSS + λ * sum(beta_j^2), λ ≥ 0

# The notation ||β||_2 demotes l_2 norm
# Define as ||β||_2 = (sum(beta_j^2))^(1/2)

# Standardizing the predictors: 
# x_ij_sp = (x_ij) * (1/n * sum(x_ij - x_j_mean)^2)^(1/2)

# Pro: bias-variance trade-off(bias but less variance)
# Con: include all p predictors, difficult for interpretation


# 6.2.1 Lasso(Least Absolute Shrinkage and Selection Operator)
# Lasso: 
# RSS + λ * sum(|beta_j|), λ ≥ 0

# The notation ||β||_1 demotes l_1 norm
# Define as ||β||_1 = sum(|beta_j|)

# Pro: produces simpler and more interpretable models
# Con: may lost information when multiple variables are effective

# Ridge regression will perform better 
# when the response is a function of many predictors, all with coefficients of roughly equal size.

# Lasso will perform better in a setting where a relatively small number of predictors have substantial coefficients, 
# and the remaining predictors have coefficients that are very small or that equal zero. 

# * Elastic Net
# combines feature elimination from Lasso and Ridge Regression.
# Elastic Net: 
# RSS + λ_2 * sum(beta_j^2) + λ_1 * sum(|beta_j|)


# 6.3 Dimension Reduction Methods

# 6.3.1 Principal Components Regression
# Principal components analysis (PCA)
# Principal Components Regression (PCR)

# 6.3.2 Partial Least Squares (PLS)


# 6.4 Considerations in High Dimensions


# 6.5 Lab: Linear Models and Regularization Methods
# 6.5.1 Subset Selection Methods

# 6.5.2 Ridge Regression and the Lasso
library(ISLR2)
dim(Hitters)
names(Hitters)
Hitters <- na.omit(Hitters)
dim(Hitters)
x <- model.matrix(Salary~., Hitters)[, -1]
head(x)
y <- Hitters$Salary
help("model.matrix")
dim(x)
length(y)

# Ridge Regression
library(glmnet)
grid <- c(10^seq(10, -2, length = 100), 0)
grid
ridge.mod <- glmnet(x, y, alpha = 0, lambda = grid)
dim(coef(ridge.mod))
help("glmnet")

# λ_2
ridge.mod$lambda[50]
coef(ridge.mod)[, 50]
sqrt(sum(coef(ridge.mod)[-1, 50]^2))

ridge.mod$lambda[60]
coef(ridge.mod)[, 60]
sqrt(sum(coef(ridge.mod)[-1, 60]^2))
predict(ridge.mod, s=50, type = "coef")[1:20,]


set.seed(1)
train <- sample(1:nrow(x), nrow(x)*0.8)
length(train)
test <- (-train)
length(test)
y.test <- y[test]
length(y.test)
c(nrow(x), length(train), length(test), length(y.test))

ridge.mod <- glmnet(x[train,], y[train], alpha = 0, 
                    lambda = grid, thresh = 1e-12)
ridge.pred <- predict(ridge.mod, s = 4, newx = x[test, ])
mean((ridge.pred - y.test)^2)

mean((mean(y[train]) - y.test)^2)

ridge.pred <- predict(ridge.mod, s = 1e-10, newx = x[test, ])
mean((ridge.pred - y.test)^2)

ridge.pred <- predict(ridge.mod, s = 0, newx = x[test, ], 
                      exact = T, x = x[train, ], y = y[train])
mean((ridge.pred - y.test)^2)

lm(y~x, subset = train)
predict(ridge.mod, s = 0, exact = T, type = "coef",
        x = x[train,], y = y[train])[1:20, ]

# Use cross-validation to choose the tuning parameter λ
set.seed(1)
length(train)
length(y.test)
cv.out <- cv.glmnet(x[train,], y[train], alpha = 0)
plot(cv.out)
bestlam <- cv.out$lambda.min
bestlam
log(bestlam)
help(cv.glmnet)

ridge.pred <- predict(ridge.mod, s = bestlam, newx = x[test, ], 
                      exact = T, x = x[train, ], y = y[train])
mean((ridge.pred - y.test)^2)

out <- glmnet(x, y, alpha = 0)
predict(out, type = "coef", s = bestlam)[1:20, ]


# The Lasso
lasso.mod <- glmnet(x[train,], y[train], alpha = 1, lambda = grid)
plot(lasso.mod)

set.seed(1)
cv.out <- cv.glmnet(x[train,], y[train], alpha = 1)
plot(cv.out)
bestlam <- cv.out$lambda.min
log(bestlam)
lasso.pred <- predict(lasso.mod, s = bestlam, newx = x[test, ])
mean((lasso.pred - y.test)^2)

out <- glmnet(x, y, alpha = 1, lambda = grid)
lasso.coef <- predict(out, type = "coef", s = bestlam)[1:20, ]
lasso.coef
lasso.coef[lasso.coef==0]
lasso.coef[lasso.coef!=0]
length(lasso.coef[lasso.coef!=0])

# 6.5.3 PCR and PLS Regression 
# Principal Components Regression
library(pls)
set.seed(2)
pcr.fit <- pcr(Salary ~ ., data = Hitters, scale = T, validation = "CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP")

set.seed(1)
pcr.fit <- pcr(Salary ~ ., data = Hitters, subset = train, 
               scale = T, validation = "CV")
validationplot(pcr.fit, val.type = "MSEP")

pcr.pred <- predict(pcr.fit , x[test , ], ncomp = 5) 
mean((pcr.pred - y.test)^2)

pcr.fit <- pcr(y ~ x, scale = TRUE , ncomp = 5) 
summary(pcr.fit)

# Partial Least Squares
set.seed(1)
pls.fit <- plsr(Salary ~ ., data = Hitters, subset = train, 
               scale = T, validation = "CV")
summary(pls.fit)

pls.pred <- predict(pls.fit , x[test , ], ncomp = 1)
mean((pls.pred - y.test)^2)

pls.fit <- plsr(Salary ~ ., data = Hitters , scale = TRUE , ncomp = 1)
summary(pls.fit)


# 6.6 Exercises
# Conceptual

# 2. 
# (a) The lasso, relative to least squares, is:
# iii. Less flexible and hence will give improved prediction accuracy when its increase in bias is less than its decrease in variance.

# (b) Ridge regression
# iii. 

# (c) non-linear methods
# ii.

# 3. Minimizing
# sum(y_i - beta_0 - sum(beta_j * x_ij))^2, 
# sum(beta_j * x_ij) ≤ s

# (a) As we increase s from 0, the training RSS will:
# iv. Steadily decreases:

# (b) for test RSS, the training RSS will:
# ii. Decrease initially, and then eventually start increasing in a U shape.

# (c) for variance, the training RSS will:
# iii. Steadily increase.

# (d) for (squared) bias, the training RSS will:
# iv. Steadily decrease.

# (e) for the irreducible error, the training RSS will:
# v. Remain constant.


# 4. Minimizing
# sum(y_i - beta_0 - sum(beta_j * x_ij))^2 + λ*sum(beta_j^2) 

# (a) As we increase λ from 0, the training RSS will:
# iii. Steadily increase.

# (b) test RSS will:
# ii. Decrease initially, and then eventually start increasing in a U shape.

# (c) variance will:
# iv. Steadily decrease.

# (d) (squared) bias will:
# iv. Steadily increase.

# (e) the irreducible error will:
# v. Remain constant.

# 5. (6.12)Ridge Regression (6.13)Lasso

# (a) p = 1, (y−β)^2 + λ*β^2; for y=2, λ=2
y <- 2
lambda <- 2
betas <- seq(-10, 10, 0.1)
func <- (y - betas)^2 + lambda * betas^2
plot(betas, func, pch = 20, xlab = "beta", ylab = "Ridge optimization")
est.beta <- y/(1 + lambda)
est.func <- (y - est.beta)^2 + lambda * est.beta^2
points(est.beta, est.func, col = "red")

# (b) p = 1, (y−β)^2 + λ*|β|; for y=2, λ=2
y = 2
lambda = 2
betas = seq(-3, 3, 0.01)
func = (y - betas)^2 + lambda * abs(betas)
plot(betas, func, pch = 20, xlab = "beta", ylab = "Lasso optimization")
est.beta = y - lambda/2
est.func = (y - est.beta)^2 + lambda * abs(est.beta)
points(est.beta, est.func, col = "red")


# Applied
# 8.
# (a)
set.seed(1)
X <- rnorm(100)
e <- rnorm(100)

# (b) 
Y <- poly(X) + e

# (e) lasso
library(glmnet)

xmat <- model.matrix(Y ~ poly(X, 10))[, -1]
lasso.mod <- cv.glmnet(xmat, Y, alpha = 1, lambda = grid)
bestlam <- lasso.mod$lambda.min
bestlam

plot(lasso.mod)



# (d)
