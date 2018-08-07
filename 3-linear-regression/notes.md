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

#### Question 1: Is there a relationship between the predictors and the response?

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

#### Question 2: Which variables are important?

- If F-statistic test indicates that at least one predictor is related to the
  response, natural follow-up: which ones?
    - Task: **variable selection**
        - Studied more extensively in Chapter 6

- Ideal test: Try out lots of models with different sets of predictors
    - Testing all combinations is computationally intesive (`q**p`)

- Three practical approaches:

1. **Forward selection**
    - Begin with _null model_ (intercept only)
    - Fit `p` regressions and add the "best" variable (lowest RSS)
    - Continue until a stopping rule is satisfied
        - e.g. Model performs below a certain RSE cutoff

2. **Backward selection**
    - Begin with all variables in the model
    - Remove the variable with the largest p-value
    - Continue until a stopping rule is satisfied
        - e.g. All remaining variables have a p-value below a certain cutoff

3. **Mixed selection** (Combine forward/backward)
    - Start with null model
    - Add the best-fit variable
    - As variables are added, p-values for in-model variables may grow; after
      a certain cutoff, remove that variable
    - Continue until a stopping rule is satisfied
        - e.g. All variables in the model are below a certain p-value cutoff,
          and all variables outside the model are above another cutoff

- Tradeoffs to each approach:
    - Forward selection
        - Pro: Is not restricted by the number of observations or predictors
        - Con: Greedy; can include predictors that later are revealed to be
          irrelevant
    - Backward selection
        - Pro: Adds the most context (less likely to include insignificant)
          variables
        - Con: Only works when `n < p` (because fitting the model won't work
          otherwise)
    - Mixed selection
        - Attempts to balance the two

#### Question 3: How well does the model fit?

- Use the two common metrics: **RSE** And **R-squared** (proportion of variance
  explained)

- In multiple regression, R-squared represents the square of the correlation
  between the response and the fitted model (`R2 = Cor(Y, Y')**2`)
    - In fact, fitted model maximizes this correlation among all possible
      models!
        - Note that this is not always desirable: more variables will always
          increase R-squared, even if the variables don't add much 
            - Comes from the fact that R-squared is computed on the training
              data
            - In contrast, RSE can _increase_ with more variables, since RSE is
              scaled by the number of predictors (`n-p-1`, simplifying to `n-2`
              in the single-variable case) and additional variables may not
              decrease RSS by enough to compensate

- Another tactic: graphical summaries
    - Useful when `p <= 3`
    - Identify areas where the model consistenly over/underperforms

#### Question 4: For an unseen value, what do we predict?

- Creating predictions from new data is trivial, but three important
  considerations should humble us:

1. The least squares plane is only an approximation of the _true population regression
   plane_ -- some amount of reducible error always remains
    - Confidence intervals help establish how close we think we should be to the
      true plane
2. The true effect is never perfectly linear, introducing some amount of _model
   bias_
3. Even if we had the population regression plane, the irreducible/random error
   `E` remains in the model

- Two ways of measuring error:
    1. **Confidence interval**: If we repeated this sampling and regression many
       times, in what interval would 95% of the scores lie?
        - Measures _reducible error_
    2. **Prediction interval**: Measures accuracy of the score for a single
       observation
        - Measures reducible error plus _irreducible error_

## 3.3 Other Considerations in the Regression Model

### 3.3.1 Qualitative Predictors

- Qualitative predictors (AKA **factors**) require some processing to make sense
    - Simplest form: two levels

#### Binary predictors

- Process: create an indicator ("dummy") variable with two possible values,
  0 and 1

```python
# Code the indicator variable. 
xi = 1 if True else 0

# Dummy variable affects the regresion like so:
if xi:
    yi = B0 + B1 + E
else:
    yi = B0 + E
```

- In this case, coefficients has the following interpretation:
    - `B0`: Average score in the falsey class 
    - `B0 + B1`: Average score in the truthy class 
    - `B1`: Average difference in scores between classes 

- However, if we code the indicators as `1` and `-1`, we can see a different
  interpretation;
    - `B0`: Overall average score
    - `B1`: "Truthy" effect

#### More than two classes

- One approach: For `n` classes, define `n-1` indicator variables as above
    - One class will use the intercept, and will be the _baseline_
    - Use F-test to test for significance

- There are more approaches, but it's beyond the scope of this book

### 3.3.2 Extensions of the Linear Model

- Two important assumptions underlying the linear model:
    - **Additive assumption**: The effect of a change in a predictor `Xj` on the
      response `Y` is independent of other predictors in `X`.
        - Assume no _interaction effects_
    - **Linear assumption**: A unit change in `Xj` produces a constant change in
      `Y`, no matter what the existing level of `Xj` is.
    - Liberman would add the _reversibility assumption_: that a negative change
      in `Xj` will lead to a corresponding negative change in `Y`

#### Removing the Additive Assumption

- Interaction effects are predicted by the `Advertising` data: the model
  consistently overpredicts sales when `TV` and `radio` are low, and
  underpredicts when they're high
    - This suggests that there's an interaction between `TV` and `radio`

- One way of dealing with interactions: introduce an **interaction term**

```python
# Standard two-variable linear model.
Y = B0 + (B1*X1) + (B2*X2) + E

# Extended with an interaction term.
Y_interaction = B0 + (B1*X1) + (B2*X2) + (B3*X1*X2) + E

# Simplify the interaction term to demonstrate the effect of X2 on X1.
Y_simplified = B0 + ((B1 + (B3*X2))*X1) + (B2*X2) + E
```

- Interpretation of `B3`: Increase (or decrease) in strength of `X1` due to
  a one-unit increase in `X2`
    - Or vice versa

- **Hierarchical principle**: If you use an interaction term you must include
  all variables in the model, _even when their coefficients are not significant_
    - Two reasons:
        1. If `X1 * X2` is predictive of the response, it doesn't matter if the
           coefficients of `X1` and `X2` are 0
        2.  `X1 * X2` is usually (always?) correlated with `X1` and `X2`

- Interaction variables have a particularly nice interpretation when quantiative
  predictors are mixed with factors
    - Note that in the absence of an interaction variable, the model defines two
      parallel lines with different intercepts:

```python
if factor == 1:
    Y = B0 + (B1*X1) + (B2*1) = B0 + (B1*X1) + B2
else:
    Y = B0 + (B1*X1) + 0 = B0 + (B1*X1)
```

- However, with an interaction variable, the model defines two lines with
  different intercepts and different slopes:

```python
if factor == 1:
    Y = B0 + (B1*X1) + (B2*1) + (B3*X1*1) = (B0 + B2) + (B1 + B3)*X
else:
    Y = B0 + (B1*X) + (B3*X1*0) = B0 + (B1*X)
```

#### Non-linear Relationships

- Simple way of extending linear regression to nonlinear situations:
  **polynomial regression**

- Intuition: To get a quick nonlinear model, use a nonlinear transformation of
  the input parameter
  - e.g. for a quadratic relationship:

```python
Y = B0 + (B1*X1) + (B2*(X1**2)) + E
```

- The key is that the above equation is still a linear model, where `X2 = X1**2`

### 3.3.3 Potential Problems

#### 1. Non-linearity of the Data

- If the true effect isn't linear, then the model will always have problems
  (bias)
    - Useful tool for figuring this out: **residual plots**
        - Simple regression: Plot the residual `ei = yi - f(xi)` versus `x`
        - Multiple regression: Plot the residual vs. the fitted `f(xi)`
            - Since there are multiple predictors
            - What does this look like?

#### 2. Correlation of Error Terms

- Certain parts of the linear model assume uncorrelated error terms
    - i.e. knowing the sign of `ei` doesn't indicate anything about the sign of
      `ei+1`
    - e.g. Standard error for coefficients
        - Correlation -> underestimate standard errors
            - p-values will overestimate confidence

- Common problem in time-series data
    - Consistent measurement errors at points in time
        - Investigate: plot the residuals vs. time
        - "tracking": adjacent residuals have similar values

#### 3. Non-constant Variance of Error Terms

- Standard error, confidence interval, and hypothesis testing rely on the
  assumption that variance among errors is constant
    - If not: funnel-shaped residual plot

#### 4. Outliers

- Generally easy to remove outliers -- harder is: principled method for
  selecting them?
    - One test: _studentized residual_
        - Residuals weighted by their estimated standard error (how many
          standard deviations the residual is away from the mean)
        - `abs(studentized_residual) > 3` -> outlier 

#### 5. High Leverage Points

- Outliers have unusual values for `y`; leverage points have unusual values for
  `x`
    - Potentially can sway the model
    - Difficult to identify in high dimensions

- Compute the **leverage statistic**
    - Always between `1/n` and 1
    - Average leverage is always `(p+1)/n`

```python
hi = (1/n) + (xi - mean(x))**2 / sum(x - mean(x)**2 for x in X)
```

#### 6. Collinearity

- Two or more predictors are closely related (correlated)
    - Makes it difficult to sort out each one's relationship with the response
        - Collinearity means that the standard error for `Bj` grows, since the
          accuracy of the coefficient estimates is lower (a wider range of
          estimates can minimize RSS)
            - Hence, t-statistic shrinks (`Bj/SE(Bj)`), and it's harder to
              reject the null
            - "low-powered test": probability of correctly detecting a non-zero
              coefficient is low

- Simple detection tool: **correlation matrix**
    - Compute the correlations between each predictor
    - Not perfect; each correlations can exist between three variables, but no
      pairwise correlations (multicollearity)

- More advanced detection tool: **variance inflation factor** (VIF) 
    - Variance of `Bj` in the full model vs. variance of `Bj` fit on its own
        - VIF > 5-10 is problematic

- Two approaches:
    1. Drop one of the problematic variables
    2. Combine collinear predictors into a single variable

## 3.5. Comparison of Linear Regression with K-Nearest Neighbors

- Parametric methods:
    - Pros: easily interpretable
    - Cons: make strong, not necessarily true assumptions about the form of the
      true effect

- **K-Nearest Neighbors regression** is closely related to the KNN classifier we
  discussed in [chapter 2](../2-statistical-learning/notes.md)
    - Recap: Given an observation `x0` and a value for `K`:
        1. Find `K` training observations closest to `x0` (the set `N0`)
        2. Estimate `f(x0)` using the average of the responses in `N0`:

```python
knn_estimate = (1/K) * (sum(y for y in N0))
```

- When is parametric better than non-parametric?
    - When the parametric form is close to the true form of `f`
    - When the true effect is substantially nonlinear, on the other hand, KNN
      can approximate the true effect much more closely
        - However: this is not guaranteed! e.g. in high dimensions with lots of
          noise (e.g. with explicit noise variables) test MSE for KNN increases
          much more quickly than linear regression
            - **Curse of dimensionality**: More dimensions -> effective
              reduction in sample size; the observations in `N0` may in fact be
              very far away from `x0` in `p`-dimensional space when `p` is large
