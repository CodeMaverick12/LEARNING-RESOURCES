
---

# Neural Networks & Machine Learning — Mathematical Notes
---

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

---

---

## 1. Neural Networks Basics

* A **neural network** takes an **input vector** and produces an **output vector** after passing through one or more **hidden layers**.
* Between the input and output layers, there can be **multiple hidden layers**.
* Each hidden layer extracts increasingly **complex features** from the input.
* The output after applying an activation function is called the **activation**.

---

## 2. Neuron Model & Forward Propagation

### Neuron Activation Formula

For neuron **j** in layer **l**:

$$
a_j^{(l)} = g\left( \mathbf{w}_j^{(l)} \cdot \mathbf{a}^{(l-1)} + b_j^{(l)} \right)
$$

Where:

* **l** = layer index
* **j** = neuron index
* **w** = weight vector
* **b** = bias
* **g** = activation function

---

## 3. Example: Hidden Layer (TensorFlow)

```python
x = np.array([[200.0, 17.0]])
layer_1 = Dense(units=3, activation="sigmoid")
a1 = layer_1(x)
```

* Double square brackets `[[ ]]` indicate a **2D matrix**.

---

## 4. Neural Network Construction (Sequential Model)

```python
layer_1 = Dense(units=3, activation="sigmoid")
layer_2 = Dense(units=1, activation="sigmoid")
model = Sequential([layer_1, layer_2])
```

 Use a **Sequential model** when data flows **linearly forward** from one layer to the next.

---

## 5. Neural Network Using NumPy

### Forward Propagation (Single Neuron)

```python
w = np.array([200, 17])
b = np.array([-1])
z = np.dot(w, x) + b
a = sigmoid(z)
```

Mathematical form:

$$
z = \mathbf{w} \cdot \mathbf{x} + b
$$

$$
a = g(z)
$$

---

## 6. Vectorized Forward Propagation

```python
def dense(A_in, W, b):
    Z = np.matmul(A_in, W) + b
    A_out = g(Z)
    return A_out
```

$$
\mathbf{Z} = \mathbf{A}_{in}\mathbf{W} + \mathbf{b}
$$

$$
\mathbf{A}_{out} = g(\mathbf{Z})
$$

---

## 7. Loss and Cost Functions

### Logistic Loss (Binary Cross Entropy)

$$
L(f(x), y) = -y \log(f(x)) - (1 - y)\log(1 - f(x))
$$

* Used for **binary classification**
* Also called **Binary Cross-Entropy Loss**

### Loss vs Cost

* **Loss function** → error for a **single example**
* **Cost function** → average loss over the dataset

$$
J = \frac{1}{m} \sum_{i=1}^{m} L(f(x^{(i)}), y^{(i)})
$$

---

## 8. Regression Loss Functions

### Mean Squared Error (MSE)

$$
J = \frac{1}{m} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
$$

### Mean Absolute Error (MAE)

$$
J = \frac{1}{m} \sum_{i=1}^{m} |y^{(i)} - \hat{y}^{(i)}|
$$

---

## 9. Backpropagation & Optimization

* Neural networks use **backpropagation** to compute gradients.
* **Gradient Descent update rule**:

$$
\theta := \theta - \alpha \frac{\partial J}{\partial \theta}
$$

### Adam Optimizer

* Adaptive learning rate
* Faster convergence
* Stable gradient updates

**Adam = Adaptive Moment Estimation**

---

## 10. Activation Functions

### Linear Activation

$$
g(z) = z
$$

✔ Used in **regression output layers**

---

### Sigmoid Activation

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

✔ Used in **binary classification output layers**

---

### ReLU (Rectified Linear Unit)

$$
g(z) = \max(0, z)
$$

✔ Used in **hidden layers**

#### Why ReLU?

* Faster than sigmoid
* Avoids vanishing gradients
* Simple computation

---

## 11. Choosing Activation Functions

* Output values **0 or 1** → **Sigmoid**
* Output values **positive and negative** → **Linear**
* Output values **only positive** → **ReLU**

 Linear activation in hidden layers cannot learn complex patterns.

---

## 12. Types of Artificial Intelligence

* **ANI (Artificial Narrow Intelligence)**
  Examples: smart speakers, self-driving cars

* **AGI (Artificial General Intelligence)**
  Can perform any task a human can do

---

## 13. Bias & Variance

### Underfitting (High Bias)

$$
J_{train} \approx J_{cv}
$$

### Overfitting (High Variance)

$$
J_{cv} > J_{train}
$$

---

## 14. Effect of Polynomial Degree

* As polynomial degree increases:

  * Training error decreases
  * Cross-validation error decreases initially
  * Then increases due to overfitting

---

## 15. Baseline Performance

* Used to estimate:

  * Human-level performance
  * Competing algorithm performance

High bias:
$$
J_{baseline} \gg J_{train}
$$

High variance:
$$
J_{cv} \gg J_{train}
$$

---

## 16. Fixing Bias & Variance

### High Variance

* Get more training data
* Use fewer features
* Increase regularization (lambda)

### High Bias

* Add more features
* Add polynomial features
* Decrease regularization (lambda)

---

## 17. Data Augmentation

* Artificially increases dataset size
* Prevents overfitting
* Improves generalization

---

## 18. Transfer Learning

1. Load pretrained model parameters
2. Fine-tune the model on your dataset

---

## 19. Evaluation Metrics

### Precision

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

### Recall

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

### F1 Score

$$
F1 = \frac{2}{\frac{1}{P} + \frac{1}{R}}
$$

---

## 20. Encoding & Sampling

* **One-Hot Encoding** converts categorical features into binary vectors
* **Sampling with replacement** creates datasets similar to the original

---

## 21. Ensemble Methods

### Boosting

* Focuses on hard-to-classify examples iteratively

### XGBoost (Extreme Gradient Boosting)

* Fast and efficient
* Built-in regularization
* Open-source implementation

```python
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
```

---

## 22. Decision Trees vs Neural Networks

* **Decision Trees**
  Best for structured (tabular) data

* **Neural Networks**
  Work on both structured and unstructured data (images, audio, text)

---


