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
