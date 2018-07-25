# Notes on Chapter 3: Linear Regression

## Introduction

- Linear regression is a good jumping-off point: many fancier techniques use
  similar intuitions

## 3.1 Simple Linear Regression

- Straightforward approach to predict a response `Y` from a single predictor `X`
    - Assumes an approximate linear relationship between `X` and `Y`
        - Mathematically:

```
Y ~= B0 + (B1*X)
```

- "Regressing `Y` on `X`"

- `B0` and `B1` are _model coefficients (or parameters)_
    - `B0`: intercept
    - `B1`: slope
    - These coefficients can predict future responses

### 3.1.1 Estimating the Coefficients

- Goal: find a line that is as close as possible to the training points
    - Problem: _how to define close_?
        - One common way: "least squares"

- Least squares regression
    - Minimize the sum of squared errors
        - "Residual sum of squares" (RSS):

```python
training = [(x1, y1), (x2, y2), ..., (xi, yi)]
RSS = sum((y - f(x))**2 for x, y in training) 
```

- Calculus gives us the following minimizers:

```python
X, Y = (x1, x2, ..., xi), (y1, y2, ..., yi)
training = zip(X, Y) 

B1 = sum((x - mean(X)*(y - mean(Y))) for x, y in training) / sum(((x - mean(X)) **2) for x in X)
B0 = mean(Y) - (B1 * mean(X))
```

- Derivation?

### 3.1.2 Assessing the Accuracy of the Coefficient Estimates

- Intuition for `B0` and `B1`:
    - `B0`: **intercept** -- expected value of `Y` for `X = 0`
    - `B1`: **slope** -- average increase in `Y` per unit increase in `X`

- `Y = B0 + B1*X + E` is the _population regression line_: the best linear
  approximation of the true relationship
  - Similar to Bayes' classifier: the ideal case, often not feasible in practice
  - Least squares line is often the best we can do, given limited data
  - Name comes from the fact that least squares is performed on a sample; if you
    had the whole population, you'd get the population regression line
    - Similar distinction as between the _sample mean_ and _population mean_
        - Sample mean is _unbiased_: if we could generate sample means over and
          over again from different samples of the population, the expected
          value would be the population mean

- "Unbiased estimator": Does not systematically over/underestimate the estimand
    - Begs the question: how accurate is the estimate, on average?
        - Answer: compute the **standard error**

#### Standard error

- Standard error for a sample mean (assuming that the `n` observations are uncorrelated):

```python
Var(u) = SE(u)**2 = (std(Y) ** 2) / n
```

- Intuitive relationship with `n`: the more samples we have, the smaller the
  error

- Standard error of linear parameter estimates (assuming that the error terms `Ei` for
  each observation are uncorrelated):

```
SE(B0)**2 = Var(E) * ((1/n) + ((mean(X)**2)/(sum((x - mean(X))**2 for x in X))))

SE(B1)**2 = Var(E) / (sum((x - mean(X))**2 for x in X))
```

- Note that `SE(B1)` is smaller when the range of `X` is larger
    - Intuitively, more values of `x` give us more leverage to estimate slope

- Note also that `Var(E)` is not known, but can be estimated with the **residual
  standard error** (derivation?):

```python
# Recall the RSS formula
RSS = sum((y - f(x))**2 for x, y in training)

RSE = sqrt(RSS /(n - 2))
```

- Looks like the mean squared residual, with two degrees of freedom...?

#### Confidence intervals

- Range of values in which we can be certain with some measure of probability
  that the true parameter value will lie in
    - Related to the standard error
    - For linear regression, 95% confidence interval (where `param` is one of
      `B0` or `B1`):

```python
confidence_int = [param - 2*SE(param), B1 + 2*SE(param)]
```

#### Hypothesis tests

- Another use for standard errors/confidence intervals

- Most common test: _null hypothesis test_
    - Are we more than 95% confident that the true relationship exists (i.e. `B1
      != 0`)?
      - Depends on the value of `SE(B1)` -- low standard error means that we can
        be more confident, in general

- One measure of confidence: **t-statistic**
    - How many standard deviations away from 0 is `B1`?
        - Formally:

```python
t = (B1 - 0) / (SE(B1))
```

- Related: **p-value**
    - What is the probability of finding a t-statistic of at least `abs(t)`? 
        - Small: It's unlikely that we could see such a difference
          between `B1` and 0 by chance
        - Large: Chances are good that we would see this value of `t` by
          accident 
    - Uses the fact that the t-distribution is well-known, and approximately
      similar to the normal distribution when `n > 30`

### 3.1.3 Assessing the Accuracy of the Model

- How well does the model fit the data?
    - Measures of "lack of fit" (or "goodness of fit")
    - Two metrics;
        1. **Residual standard error (RSE)**
        2. **R-squared**

#### Residual standard error

- Estimate of the standard deviation of E, the irreducible error term
    - How much does the average response (data point) differ from what is predicted by the
      true effect?

- Once again:

```python
RSE = sqrt(RSS/(n-2))
```

- Reported in the same units as the response

#### R-squared

- As opposed to RSE, R-squared is a unit-less proportion
    - In some sense, a more "objective" (standardized) measure of goodness of
      fit
    - AKA: "proportion of variance explained"

- Mathematically:

```python
# Total sum of squares -- sum of squared difference from the mean
TSS = sum((y - mean(y))**2 for _, y in training)

# Residual sum of squares -- sum of squared difference from the observation
RSS = sum((y - f(x))**2 for x, y in training)

# R-squared
R2 = (TSS - RSS)/TSS = 1 - (RSS/TSS)
```

- Think of TSS as the inherent variability in responses `Y`, before regression
    - RSS, on the other hand, represents the variability that remains _after_
      regression
        - Hence, R-squared measures: how much of the inherent variability (TSS)
          is explained by the regression variability (RSS)?

- In single linear regression, R-squared is equivalent to the **squared
  correlation**
    - However, this doesn't hold in higher dimensions, since correlation is
      a pairwise comparison
        - R-squared serves as a replacement

## 3.2 Multiple Linear Regression

- Extend linear regression for higher dimensions such that each predictor gets
  its own slope coefficient:

```python
Y = B0 + (B1*X1) + (B2*X2) + ... + (Bp*Xp) + E
```

- Interpretation of coefficients: **average effect on Y of a one-unit increase
  in Xj, holding all other predictors fixed**
    - Last part is important! Assumes independence between the predictors. If
      predictors are at all correlated, it's not meaningful to "hold them fixed"

### 3.2.1 Estimating the Regression Coefficients

- As in simple linear regression: minimize sum of squared residuals (RSS)

- Simple/multiple regression coefficients can be quite different!
    - In particular: if two predictors are correlated, one of them may appear
      very significant in a single regression setting, but have little to no
      effect under multiple regression
        - e.g. Shark attacks, ice cream sales, and temperature

### 3.2.2 Some important questions

- Four questions implicated by multiple regression:

1. Is at least one of the predictors useful?
2. Do all the predictors help explain `Y`, or only a subset?
3. How well does the model fit the data?
4. Given a set of predictor values, what would we predict?

#### Is there a relationship between the predictors and the response?

- Extend hypothesis testing to multiple variables
    - Hypothesis: All the coefficients are 0

- **F-statistic**
    - Is at least one predictor useful in predicting `Y`?
    - When F-statistic is close to 1, hypothesis cannot be rejected
        - How close? Use values of `n` (number of observations) and `p` (number
          of predictors) to compute the p-value
            - Note: F-statistic can be shown to follow an F-distribution when
              the error term `E` is normally distributed, and/or `n` is very
              large

- Formally:

```python
# Compute the F-statistic
F = ((TSS-RSS)/p) / (RSS/(n-p-1))
```

- Both numerator and denominator have an expected value of `Var(E)` when the
  null hypothesis is true, so in that case we expect `F = 1` 

- How to compute F-statistic for a subset of the predictors?
    - Substitute `TSS` for `RSS0`
        - Where `RSS0` represents the residual sum of squares for a model that
          is missing the predictors in question

```python
# Compute the F-statistic for a subset of predictors
# (q is the number of predictors in the subset)
F = ((RSS0-RSS)/q) / (RSS/(n-p-1))
```

- t-statistic is equivalent to the F-statistic for the subset of the data that
  only includes one predictor
    - Square of the t-statistic is the corresponding F-statistic
        - Derivation? (This would be fun to show algebraically)
    - "Partial effects"

#### A note on p-hacking

- In the case where there are a lot of predictors and the null hypothesis is
  true, a certain percentage of p-values will be below 0.05 by chance (the
  nature of the t-distribution)
    - e.g. when `p = 100`, we should expect 5% of p-values to be below 0.05
      (since t-distribution approximates normal distribution)

- F-statistic is relatively resilient to this problem, since it factors in the number of
  predictors (degrees of freedom)
    - However, when `p` is very large (in particular when `p > n`), the
      F-statistic and linear regression do not work, and we need to rely on
      different methods
