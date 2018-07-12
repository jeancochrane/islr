# Exercises for Chapter 2

## Conceptual

### Question 1

For each of parts (a) through (d), indicate whether we would generally
expect the performance of a flexible statistical learning method to be better or
worse than an inflexible method. Justify your answer.

a. The sample size `n` is extremely large, and the number of predictors `p` is
   small.

b. The number of predictors `p` is extremely large, and the number of
   observations `n` is small.

c. The relationship between the predictors and response is highly non-linear.

d. The variance of the error terms, i.e. `std^2 = Var(E)`, is extremely high.

### Answer 1

a. Better. With a large sample size, flexible models are more likely to
   approximate the true effect.

b. Worse. With a low sample size, a simpler model is safer to prevent
   overfitting.

c. Better. Flexible models permit nonlinearity.

d. Worse. In this case, a flexible model may accidentally end up fitting to 
   the irreducible error.

### Question 2

Explain whether each scenario is a classification or regression problem, and
indicate whether we are most interested in inference or prediction. Finally,
provide `n` and `p`.

a. We collect a set of data on the top 500 firms in the US. For each firm we
record profit, number of employees, industry, and the CEO salary. We are
interested in understanding which factors affect CEO salary.

b. We are considering launching a new product and wish to know whether it will
be a success or failure. We collect data on 20 similar products that were
previously launched. For each product we have recorded whether it was a success
or failure, price charged for the product, marketing budget, competition price,
and ten other variables.

c. We are interested in predicting the percent change in the USD/Euro exchange
rate in relation to the weekly changes in the world stock markets. Hence we
collect weekly data for all of 2012. For each week we record the percent change
in the USD/Euro, the percent change in the US market, the percent change in the
British market, and the percent change in the German market.

### Answer 2

a. Regression (CEO salary), inference, `n = 500`, `p = (profit, employees,
industry)``.

b. Classification (success or failure), prediction, `n = 20`, `p = (price, budget,
competition price, +10 variables)``.

c. Regression (percent change in exchange rate), prediction, `n = 52`, `p = (%
change in US market, % change in British market, % change in German market)``.

### Question 5

What are the advantages and disadvantages of a very flexible (versus a less
flexible) approach for regression or classification? Under what circumstances
might a more flexible approach be preferred to a less flexible approach? When
might a less flexible approach be preferred?

### Answer 5

In general, flexibility incurs the bias-variance trade-off: less flexible models
are higher in bias but lower in variance, while more flexible models tend to
incur more variance and less bias. More flexible models are also prone to
overfitting, particularly when the training data has a small sample size or is
very noisy. Finally, the more flexible a model is, the harder it is to
interpret, so flexible models tend to be more useful for prediction than
inference.

Some cases where flexible models perform better:

- Large sample sizes
- Low noise
- An underlying nonlinear model
- A task where prediction is the goal, rather than inference 

Some cases where inflexible models perform better:

- Small sample sizes
- Lots of noise
- An underlying linear model
- A task where inference is the goal

### Question 6

Describe the differences between a parametric and a non-parametric staistical
learning approach. What are the advantages of a parametric approach to
regression or classification (as opposed to a non-parametric approach)? What are
the disadvantages?

### Answer 6

Parametric approaches (like linear regression) assume that an underlying effect is some sort of linear
combination of parameters and input variables, and seeks to estimate the
parameters for the effect. Non-parametric approaches (like K-means clustering)
assume no such underlying effect, and seek to make estimates in other ways.

Parametric approaches tend to be useful in inference and regression problems, since their
results are highly interpretable and allow a researcher to say particular things
about the effect of each input variable on the overall model. Non-parametric
approaches, on the other hand, are useful in prediction and clustering/classification
problems, since they are not limited by any particular model form and so can
more flexibly fit a wider variety of data. 

Non-parametric approaches also require much larger sample sizes to avoid overfitting the training data.

Parametric approaches require a theory of the underlying effect, or else they
risk introducing bias or overfitting.

### Question 7

The table below provides a training data set containing six observations, three
predictors, and one qualitative response variable.

obs | X1 | X2 | X3 |   Y   |
--- | -- | -- | -- | ----- |
1   | 0  | 3  | 0  | red   |
2   | 2  | 0  | 0  | red   |
3   | 0  | 1  | 3  | red   |
4   | 0  | 1  | 2  | green |
5   | -1 | 0  | 1  | green |
6   | 1  | 1  | 1  | red   |

Suppose we wish to use this data set to make a prediction for `Y` when `X1 = X2
= X3 = 0` using K-nearest neighbors.

a. Compute the Euclidean distance between each observation and the test point,
`X1 = X2 = X3 = 0`.

b. What is our prediction with `K = 1`? Why?

c. What is our prediction with `K = 3`? Why?

d. If the Bayes decision boundary in this problem is highly non-linear, then
would we expect the _best_ value for K to be large or small? Why?

### Answer 7

a. To compute Euclidean distances, we can define the following R function:

```r
dist = function(x1, x2) sqrt(sum((x1 - x2) ^ 2))
```

The distances are then:

```r
> dist(c(0, 0, 0), c(0, 3, 0))
[1] 3

> dist(c(0, 0, 0), c(2, 0, 0))
[1] 2

> dist(c(0, 0, 0), c(0, 1, 3))
[1] 3.162278

> dist(c(0, 0, 0), c(0, 1, 2))
[1] 2.236068

> dist(c(0, 0, 0), c(-1, 0, 1))
[1] 1.414214

> dist(c(0, 0, 0), c(1, 1, 1))
[1] 1.732051
```

b. With `K = 1`, the nearest point is observation 5, so our prediction is `Green`.

c. With `K = 3`, the three nearest points are observations 2, 5, and 6. Hence,
the probabilities of the two classes are:

```
prob(Green) = 1/3
prob(Red) = 2/3
```

So our prediction is `Red`.

d. When the Bayes decision boundary is highly non-linear, we generally expect
smaller values for K to produce better results. This is because smaller values
of K on average are more flexible.
