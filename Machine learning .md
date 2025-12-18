

---

# Machine Learning Notes

---

## 1. Introduction

1. Machine Learning (ML) is the field that gives computers the ability to **learn without being explicitly programmed**.
2. Types of ML algorithms:

   * Supervised Learning
   * Unsupervised Learning
   * Recommender Systems
   * Reinforcement Learning

---

## 2. Supervised Learning

1. Supervised ML algorithms **learn to predict `x` and `y` values**.
2. **Classification models** predict **categories** (non-numeric).
3. Supervised models **learn from labeled data** (given correct answers).

### Regression vs Classification

* **Regression** → Predict numeric values
* **Classification** → Predict categories

---

## 3. Unsupervised Learning

1. Data only comes with inputs (`X`) but **no labels (`Y`)**.
2. Algorithms must **find structure in the data**.
3. Types:

   * **Anomaly Detection** → Find unusual data points
   * **Dimensionality Reduction** → Compress large data into smaller sets

---

## 4. Linear Regression

1. Linear model: `f = w*x + b`

   * Parameters `w` and `b` can be adjusted to improve model performance
2. Cost function measures **error**:

[
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}^i - y^i)^2
]

* Lower cost → Better predictions
* Objective: **Minimize cost** through `w` and `b`

---

### Gradient Descent

```python
w = w - alpha * dJ/dw
b = b - alpha * dJ/db
```

* `alpha` = Learning rate (step size)
* If `alpha` is too large → Overshoot minimum
* If `alpha` is too small → Slow convergence
* Local minima → slope = 0 → parameters stop changing
* Convex function → Single global minimum

---

### Multiple Linear Regression

* Use **vectorization** for efficiency:

```python
np.dot(w, X)
```

* **Normal Equation** → Solve for `w` and `b` without iterations

---

### Feature Scaling

* Enhances **gradient descent speed**

* Methods:

  1. **Min-Max Scaling:** (x_{scaled} = x / x_{max})
  2. **Mean Normalization:** (x_{scaled} = (x - \mu) / (x_{max} - x_{min}))
  3. **Z-score Normalization:** (x_{scaled} = (x - \mu) / \sigma)

* Verify gradient descent: Plot **iterations vs cost function**

---

## 5. Logistic Regression

1. Sigmoid function:

[
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = w \cdot x + b
]

* Output between 0 and 1
* Decision boundary can be **non-linear**
* Cost function is **non-convex**, so linear regression cost cannot be used

2. Logistic Loss Function:

[
L(f(x^i), y^i) =
\begin{cases}
-\log(f(x^i)) & \text{if } y^i = 1 \
-\log(1 - f(x^i)) & \text{if } y^i = 0
\end{cases}
]

* Loss → Single training example
* Cost → Full training set

---

## 6. Polynomial Features & Overfitting

* Higher-degree polynomial → Better fit but may **overfit**
* Low-degree polynomial → Good balance
* **Regularization** → Reduces parameter values to prevent overfitting

[
J(w,b) = \frac{1}{2m} \sum (f(x^i)-y^i)^2 + \frac{\lambda}{2m} \sum w_j^2
]

* `lambda` = Regularization parameter

  * Too large → Underfit
  * Zero → Overfit

---

## 7. Entropy

* **Entropy** measures **impurity** in data


