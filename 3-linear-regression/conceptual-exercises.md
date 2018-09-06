# Conceptual Exercises for Chapter 3

## Question 1

Describe the null hypotheses to which the p-values given in Table 3.4
correspond. Explain what conclusions you can draw based on these p-values. Your
explanation should be phrased in terms of `sales`, `TV`, `radio`, and `newspaper`, rather than in terms of the coefficients of the linear model.

## Answer 1

For ease of reference, here is a reproduction of Table 3.4:

| Variable    | Coefficient | Std. error | t-statistic | p-value  |
| ----------- | ----------- | ---------- | ----------- | -------- |
| `Intercept` |       2.939 |     0.3119 |        9.42 | < 0.0001 |
| `TV`        |       0.046 |     0.0014 |       32.81 | < 0.0001 |
| `radio`     |       0.189 |     0.0086 |       21.89 | < 0.0001 |
| `newspaper` |      -0.001 |     0.0059 |       -0.18 |   0.8599 |

For each variable, the null hypothesis is similar: "This variable has no effect
on sales, _when considered in the context of all the other variables_". This interpretation is rather straightforward, since the null hypothesis simply states that the
advertising type in question has no effect on sales when considered alongside all of
the other advertising types.

Given the p-values in Table 3.4, we can reject the null hypothesis in the case
of `TV` and `radio` (i.e. both TV and radio ads have a significant effect on
sales in a market containing all advertising types), but we cannot reject the
null hypothesis in the case of `newspaper` (i.e. there is a reasonable chance
that newspaper ads have no effect on sales in a market containing all
advertising types).

## Question 2

Carefully explain the differences between the KNN classifier and KNN regression
methods.

## Answer 2

The difference between KNN as a classifier and as a regression method is the
same as the difference between classification and regression tasks in general:
classification tasks seek to predict a qualitative (factor) label for an
observation, while regression tasks seek to predict a quantitative label for an
observation.

For example, if your goal was to predict whether a neighborhood in Boston had a housing stock 
of the age `new`, `medium`, or `old` (qualitative) based on the median home value (quantitative), you might
use a KNN classifier; however, if you instead wanted to predict the median home
value of the neighborhood based on the age of its housing stock, you might use
KNN regression.

## Question 3

Suppose we have a data set with five predictors, `X1` (GPA), `X2` (IQ score),
`X3` (Gender, 1 for Female and 0 for Male), `X4` (interaction between GPA and IQ), and `X5` (interaction between GPA and Gender). The response is starting salary after graduation (in thousands of dollars). Suppose we use least squares to fit the model, and get `B0 = 50`, `B1 = 20`, `B2 = 0.07`, `B3 = 35`, `B4 = 0.01`, `B5 = -10`.

### Question 3a.

Which answer is correct, and why?

i. For a fixed value of IQ and GPA, males earn more on average than females.
ii. For a fixed value of IQ and GPA, females earn more on average than males.
iii. For a fixed value of IQ and GPA, males earn more on average than females
provided that the GPA is high enough.
iv. For a fixed value of IQ and GPA, females earn more on average than males
provided that the GPA is high enough.

### Answer 3a.

`iii` above is the correct answer: holding IQ and GPA constant, given a high enough GPA, men earn more on
average than women in this dataset.

To see why, consider the coefficients related to gender. `B3` is the Gender
coefficient -- since women are coded as `1` and men as `0`, the positive sign on
this coefficient indicates that there is a salary premium for women.

However, this salary premium alone doesn't tell the whole story. Note that the
coefficient for the _interaction term between GPA and gender_, `B5`, is both
sizable and negative. Since women are coded as `1`, this interaction term will
appear in the model only for women, and will be scaled by a factor according to
their GPA. This tells us that when GPA >= 3.5, the effect of the interaction
between GPA and gender will entirely cancel out the salary premium for women
(`35`), and beyond that GPA men will start to earn more than women holding GPA
and IQ constant.

### Question 3b.

Predict the salary of a female with IQ of 110 and a GPA of 4.0.

### Answer 3b.

Given the coefficients above, the corresponding model should be:

```python
salary = B0 + (B1*gpa) + (B2*iq) + (B3*gender) + (B4*(gpa*iq))
+ (B5*(gpa*gender))
       = 50 + (20*gpa) + (0.07*iq) + (35*gender) + (0.01*(gpa*iq)) + (-10*(gpa*gender))
       = 50 + (20*4.0) + (0.07*110) + (35*1) + (0.01*(4.0*110)) + (-10*(4.0*1))
       = 137.1
```

Assuming the result is in units of $1000/year, we predict a salary of $137,100 per year.

### Question 3c.

True or false: Since the coefficient for the GPA/IQ interaction term is very
small, there is very little evidence of an interaction effect. Justify your
answer.

### Answer 3c.

This question is worded ambiguously, so I'll give an equally ambiguous answer:
"it depends on the confidence interval of the interaction coefficient, which is not
available to us here." If the confidence interval is large, there is a chance
that the actual interaction effect is large and we simply didn't observe it out
of chance; if the confidence interval is very small, on the other hand, then the
small interaction term is a strong indication that there is not a major
interaction effect between GPA and IQ.

## Question 4

I collect a set of data (n = 100) containing a single predictor and
a quantitative response. I then fit a linear regression model to the data, as
well as a separate cubic regression, i.e. `Y = B0 + B1*X1 + B2*X2^2 + B3*X3^3
+ E`.

### Question 4a.

Suppose that the true relationship between X and Y is linear, i.e. `Y = B0
+ B1*X1 + E`. Consider the training residual sum of squares (RSS) for the linear
  regression, and also the training RSS for the cubic regression. Would we
  expect one to be lower than the other, would we expect them to be the same, or
  is there not enough information to tell? Justify your answer.

### Answer 4a.

Even though the true relationship is linear, I wouldn't be surprised if the
training RSS is lower for the cubic regression, since it would have a better
chance of overfitting the data and modelling the irreducible error.

### Quesiton 4b.

Answer (a) using test rather than training RSS.

### Answer 4b.

In the case of test RSS, we would expect the linear model to have a lower RSS
than the cubic model, since the cubic model would overfit the irreducible
error and not reliably model the true effect.

### Question 4c.

Suppose that the true relationship between X and Y is not linear, but we don't
know how far it is from linear. Consider the training RSS for the linear
regression, and also the training RSS for the cubic regression. Would we expect
one to be lower than the other, would we expect them to be the same, or is there
not enough information to tell? Justify your answer.

### Answer 4c.

Even though we don't know the precise form of the true effect, we can be
confident that the cubic model will perform better on the training data (have
a lower training RSS) than the linear model, since the cubic model is more
flexible in general.

### Question 4d.

Answer (c) using test rather than training RSS.

### Answer 4d.

There's not enough information to know whether the linear model will have
a lower test RSS than the cubic model. If the true effect is closer to linear
than cubic, the cubic model will overfit the data; if it's closer to cubic than
linear, however, the cubic model will be more flexible.

## Question 6

Using (3.4), argue that in the case of simple linear regression, the least
squares line always passes through the point `(mean(x), mean(y))`.

## Answer 6

Equation (3.4) states that the least squares line can be written as:

```python
y = mean(Y) - B1*mean(X) + B1*x
```

It follows that the point `(mean(X), mean(Y))` satisfies this equation for all values of `B1`:

```python
mean(Y) = mean(Y) - B1*mean(X) + B1*mean(X)
mean(Y) = mean(Y) - 0
0 = 0
```
