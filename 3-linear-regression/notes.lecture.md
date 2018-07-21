# Notes on Chapter 3: Linear Regression

## Course lecture notes

### Least squares regression, standard error, and confidence intervals

- True effects are never linear, but linear models can be a good approximation
    - Especially if there's lots of noise around an almost-linear true effect,
      in which case a flexible model might overfit to the noise
    - Simplicity is useful! Both for explanatory power, and also for helping us
      get a handle on foundations

- Regression questions:
    - Any relationship?
    - How strong of a relationship?
    - Which predictors?
    - How accurate are predictions?
    - Linear?
    - Synergy between predictors?

- Defining linear regression
    - **Minimizes residual sum of squares** of the training data
        - "Least squares"

- Assessing accuracy of coefficients
    - **Standard error**: how much would the coefficient estimate change if we
      took different samples?
        - Standard error of the slope (`B1`): variance of the noise, divided by
          the sum of squared difference between `xi` and the mean `x`
            - Intuition: If noise is very high, slope is unreliable; on the
              other hand, the wider the range of `x` values is, the better the
              approximation of the slope will be

- Confidence interval for approximation of `B1`
    - Under conditions of repeated sampling: 95% probability that the true value
      of `B1` is contained in the interval:

```
[B'1 - (2*SE(B'1)), B1 + (2*SE(B'1))]
```

### Hypothesis testing

- Closely related to confidence intervals for parameter approximation
    - Most common type: null-hypothesis testing
        - How likely is it that there is a relationship between `X` and `Y`?
            - i.e. How likely is it that `B1 != 0`?
        - Testing null hypothesis: **t-statistic**
            - Definition: estimated slope divided by the standard error
            - p-value: Probability of getting a t-value at least as high as you
              got
                - Assumes large, normally-distributed samples (?)
        - Relationship to confidence intervals: if we reject the null hypothesis
          (i.e. the hypothesis test "fails") then the 95% confidence interval
          for the slope coefficient `B1` will not contain 0 

### Assessing model accuracy

- Confidence interval assesses accuracy of the slope coefficient, but what about
  accuracy of the model altogether?
    - **Residual standard error**: Mean squared residual
    - **R-squared**, or fraction of the variance explained (`1 - (TSS/RSS)`,
      where TSS is the sum of squared differences from the mean)

        - Intuition: How much of an improvement did we make over the naive case
          (just using the mean as an estimate)?
        - But what does this have to do with explaining variance?
        - Equivalent to the squared correlation between `X` and `Y` (hence,
          "R-squared")

### Multiple linear regression

- Slope (`Bj`) for each predictor in the model
    - Intuition: the _average_ effect on `Y` for a one-unit increase in `Xj`,
      _holding all other predictors constant_

- Line -> hyperplane

- Interpreting multiple regression
    - Ideal case: predictors are uncorrelated (a "balanced design")
        - In this case, we can safely interpret the coefficients as representing
          the average effect on `Y`
    - Correlations cause problems
        - Increased variance of coefficients
        - Changing a predictor causes changes in other predictors
    - Avoid causal interpretation of observational data using linear regression

- George Box: "Essentially, all models are wrong, but some are useful"

### Estimation/Prediction for Multiple Regression

- Similar problem: minimize the sum of the squared residuals
