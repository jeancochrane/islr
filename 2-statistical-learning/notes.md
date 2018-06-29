# Notes on Chapter 2: Statistical Learning

## 2.1: What Is Statistical Learning?

- Basics: predict **output variables** from **input variables**
    - Input variables usually denoted by X{subscript}
        - Other names: *predictors*, *independent variables*, *features*
    - Output variable(s) denoted by Y{subscript}
        - Other names: *response*, *dependent variable*

- General model: Y = f(X) + E
    - Where X = set of predictors (X1, X2, ..., Xp)
    - E is a random **error term**
        - Independent of X with a zero mean
    - f is the "systematic" information that X provides about Y

- However, we're not just restricted to two variables (X and Y)!
    - e.g. two features -> 3d surface

- "Statistical learning": **approaches for approximating f**

### 2.1.1 Why Estimate f?

- Two reasons: **prediction** and **inference**
    - Are these not the same thing?

#### Prediction

- Why? We often have inputs but no reliable output; f' is a way of
  approximating that output
    - e.g. House prices based on location, house size, age, etc.

- In this case, f' is a "black box": we don't really care what form it takes, as
  long as it produces reliable predictions

- Accuracy depends on **reducible** and **irreducible error**
    - Reducible: The model just isn't good enough yet; keep trying!
    - Irreducible: The intrinsic randomness of the process (error term E)
        - E may contain unmeasured variables that influence the explanadum;
          E may also include variation that we can't measure directly
            - But wouldn't the former be a reducible error...?
    - Irreducible error provides a limit on how accurate we can get

- More formally:

```
E(Y - Y')^2 = E[f(X) + e - f'(X)]^2 = [f(X) - f'(X)]^2 + Var(e)
```

Where `[f(X) - f'(X)]^2` is the reducible error, and `Var(e)` (the variance of e) 
is the irreducible error.
    - Derivation?

#### Inference

- Why? We often want to know how Y will change as features change
    - e.g. We want to know employment changes as a function of the minimum wage
    - Implicit intervention: If a feature X were to change, how would Y respond?

- In this case, f' is not a black box: it represents a theory about the world
    - Answer questions like:
        - Which features are relevant (have the most power)?
        - How do different features affect the direction of the delta of Y?
        - Is the relationship linear, or more complex?
    - As a theory, sometimes simpler models may be more desirable, even though
      their predictive accuracy for a given data set is sub-optimal 
        

- Often, we're interested in both prediction and inference

### 2.1.2 How Do We Estimate f?

- Shared characteristics of all the approaches we'll cover:
    - Observe a sample of count `n` observations
        - Call these "training data"
    - Apply a statistical learning method to training data to estimate f

- Two types of approaches: **parametric** and **non-parametric**

#### Parametric methods

- Estimating f is basically estimating a set of parameters (or weights) for X
    - Usual workflow:
        1. Assume a certain kind of model (e.g. linear: Y is a linear
           combination of X)
        2. Find a parameter vector B that estimates X from Y (i.e. "fit" the
           model) with a procedure

- Pros: simplifies the process, since estimating linear parameters B is easier than
  fitting an arbitrary `p`-dimensional function
    - What does "arbitrary `p`-dimensional function" mean in this case? That
      inputs X behave nonlinearly, perhaps?

- Cons:
    - Model will not accurately model `f`
        - Linear relationships are rare in the wild
    - Danger of overfitting the data
        - Wouldn't this be worse if we were estimating arbitrary functions
          though?

#### Non-parametric methods

- No assumption about the "functional form" of `f`
    - Just fit the data as close as possible, within reasonable constraints

- Pro: can approximate `f` much more closely, since no particular functional
  form is required

- Con: much more training data is required for it to make sense

### 2.1.3 The Trade-Off Between Prediction Accuracy and Model Interpretability

- Models vary in their "flexibility": i.e. the range of different shapes that
  they allow `f'` to take
    - e.x. linear models (like linear regression) are very inflexible

- Inflexible models often offer the benefit of interpretability
    - This is useful in cases where we're trying to perform inference (where we
      want to understand the true effect)
        - In the case of linear models, we have very explicit "weights" that are
          associated with particular input features; this tells a clear and
          compelling story about features in isolation of one another

- Flexible models, on the other hand, are harder to interpret, and the
  relationship between their features and the observed Y will be less clear

- I see why models that permit complex, polynomial interactions between
  features (e.g. something like `Y = (2x1 + x2)^3 + x2`) would make it harder to
  parse the impact of individual features, but I don't see why this is
  necessarily the case for e.g. generalized additive models, which extend the
  linear model to allow some non-linear relationships (the relationship between
  each feature and the outcome can be modeled as a curve, instead of a straight
  line). Is there something about linearity in particular that makes it an easier
  relationship to interpret, even outside of interactions between features?

- Even when prediction (and not inference) is the goal, however, sometimes less
  flexible models perform better because more flexible models are more prone to
  overfitting

### 2.1.4 Supervised vs. Unsupervised Learning

- Two main categories of statistical learning problems: supervised and
  unsupervised

- Supervised methods: the category represented by every approach we've
  discussed so far 
  - Given a response `Yi` for each observation `Xi`, what is the most accurate
    model that predicts `Y` for future observations, or that clarifies the relationship
    between `Y` and features of `X`?

- Unsupervised methods: observe a vector of measurements `Xi`, without the
  corresponding response `Yi`
  - Possible questions: how are the observations related? Are there natural
    "groups" that we can define among them? How would we group future
    observations?
    - "Clustering" problems

- "Semi-supervised" problems: situations where both approaches are appropriate
    - e.g. if we have some labeled observations, but also some unlabeled (if
      labels are expensive to collect, say)
      - Beyond the scope of the book :(

### 2.1.5 Regression vs. Classification

- **Regression** problems tend to involve quantitative variables (numbers)

- **Classification** problems tend to involve categorical variables (classes) 

- Distinction is not clean-cut
    - E.g.: logistic regression estimates probabilities (like a regression), but
      for categorical classes
        - Don't all methods, regression and classification alike, estimate
        probabilities? What defines a "regression" precisely?
    - E.g.: KNN and boosting can be used in the case of both quantitative and
      qualitative responses

- The appropriate type of method (regression or classification) depends more on
  the form of the outcome rather than the form of the features, since
  categorical features can always be coded
  - Perhaps a better distinction than quantitative/qualitative is
    continuous/discrete?

## 2.2 Assessing Model Accuracy

- Key insight: different methods work well in different contexts
    - Specific to the dataset and the task at hand

### 2.2.1 Measuring the Quality of Fit

- Goal: quantify how close the predictions are to the true response

- Most common technique in **regression** is **mean squared error (MSE)**. In
  Pythonic pseudocode: 

```python
# Pairs of observations (xi) with responses (yi)
observations = [(x1, y1), (x2, y2), ..., (xn, yn)]

# Assuming a model `f`
MSE = (1 / len(observations)) * sum(((y - f(x)) ** 2) for (x, y) in observations)
```

- Since MSE above is computed using training data, it is technically the
  *training MSE*
  - More interesting: how does the model perform on data it hasn't seen yet?
    - "test MSE"

- Synonym for flexibility in a model: **degrees of freedom**
    - Training MSE declines monotonically as degrees of freedom increase; test
      MSE, however, does not
        - Sign of overfitting
