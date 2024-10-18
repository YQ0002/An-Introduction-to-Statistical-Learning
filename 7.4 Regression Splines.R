rm(list = ls())
# file:///D:/a1%E6%96%87%E4%BB%B6/2023%E5%AF%8C%E5%B1%B1/ISLRv2_corrected_June_2023.pdf
# P-295

# Regression Splines

# Lab
# 7.8.2 Splines
library(ISLR2)
library(splines)
attach(Wage)
fit <- lm(wage~bs(age, knots=c(25,40,60)), data = Wage)

# Create a grid of values for age
age.grid <- seq(min(age), max(age))
pred <- predict(fit, newdata=list(age=age.grid), se=T)

# Plot
plot(age, wage, col = "gray")
lines(age.grid, pred$fit, lwd = 2)
lines(age.grid, pred$fit + 2 * pred$se, lty = "dashed")
lines(age.grid, pred$fit - 2 * pred$se, lty = "dashed")

dim(bs(age, knots = c(25, 40, 60)))
dim(bs(age, df = 6))
attr(bs(age, df = 6), "knots")

# natural spline: ns() function
fit2 <- lm(wage~ns(age, df = 4), data = Wage)
pred2 <- predict(fit2, newdata=list(age=age.grid), se=T)
lines(age.grid, pred2$fit, col = "red", lwd = 2)

# smooth spline: smooth.spline() function
plot(age, wage, xlim = range(age), cex=0.5, col="darkgrey")
title("Smoothing Spline")
fit <- smooth.spline(age, wage, df=16)
fit2 <- smooth.spline(age, wage, cv = T)
fit2$df
# 6.794596
lines(fit, col = "red", lwd = 2)
lines(fit2, col = "blue", lwd = 2)
legend("topleft", legend = c("16 DF", "6.8 DF"),
       col = c("red", "blue"), lty = 1, lwd = 2, cex = .8)


# local regression: loess() function
plot(age, wage, xlim = range(age), cex=0.5, col="darkgrey")
title("Local Regression")
fit <- loess(wage~age, span = 0.2, data = Wage)
fit2 <- loess(wage~age, span = 0.5, data = Wage)
lines(age.grid, predict(fit, data.frame(age = age.grid)), col = "red", lwd = 2)
lines(age.grid, predict(fit2, data.frame(age = age.grid)), col = "blue", lwd = 2)
legend("topright", legend = c("Span = 0.2", "Span = 0.5"),
       col = c("red", "blue"), lty = 1, lwd = 2, cex = .8)
# The larger the span, the smoother the ft





# Polynomial Regression and Step Functions
fit1 <- lm(wage~poly(age, 4), data = Wage)
summary(fit1)

fit2 <- lm(wage~poly(age, 4, raw = T), data = Wage)
summary(fit2)

# create a grid of values for age
age.grid <- seq(min(age), max(age))
preds <- predict(fit1, newdata = list(age=age.grid), se=T)
se.bands <- cbind(preds$fit + 2*preds$se.fit,
                  preds$fit - 2*preds$se.fit)
# plot
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 1, 1),
    oma = c(0, 0, 4, 0))
plot(age, wage, xlim = agelims, cex = .5, col = "darkgrey")
title("Degree -4 Polynomial", outer = T)
lines(age.grid, preds$fit, lwd = 2, col = "blue")
matlines(age.grid, se.bands, lwd = 1, col = "blue", lty = 3)

preds2 <- predict(fit2, newdata = list(age=age.grid), se=T)
max(abs(preds$fit - preds2$fit))














hist(age)
summary(lm(wage~age))
# Cubric Splines

#cubic spline
lm(wage ~ age + I(age^2) + I(age^3) +
     I((age-20)^3*(age>=20)) +
     I((age-30)^3*(age>=30)) +
     I((age-40)^3*(age>=40)) +
     I((age-50)^3*(age>=50)) +
     I((age-60)^3*(age>=60)), 
   data = Wage)

# b-spline
library(splines)
lm(wage ~ bs(age, knots = c(20,30,40,50,60)),data = Wage)


plot(age, wage)
abline(lm)

age_grid <- seq(min(age), max(age))
pred1 <- predict(fit1, newdata = list(age=age_grid))
lines(age_grid, pred1, type = "l")

