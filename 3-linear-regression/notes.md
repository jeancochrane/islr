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
