# Task — 16: Neural Networks with Keras & Training Deep Networks

> **Goal:** Go from a single artificial neuron to a well-trained deep network: understand the Perceptron, the Multilayer Perceptron and backpropagation, build and tune models with the Keras API, then learn the techniques that make deep networks actually train — smart initialization, better activations, Batch Normalization, faster optimizers, learning-rate schedules, and regularization (dropout and friends).

> **Note:** This continues the ML track (Task 10 → Task 16). Tasks 14 and 15 already used neural networks (NeuMF, two-tower, softmax DNNs) as building blocks. Task 16 opens those boxes: what a layer computes, how gradients reach every weight, and why a deep model that "should" work often won't train until you apply the right tricks.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [What You Will Learn](#2-what-you-will-learn)
3. [Prerequisites & Setup](#3-prerequisites--setup)
4. [Phase 1 — From Biological Neurons to the Perceptron](#phase-1--from-biological-neurons-to-the-perceptron)
5. [Phase 2 — The Multilayer Perceptron, Backpropagation & Activation Functions](#phase-2--the-multilayer-perceptron-backpropagation--activation-functions)
6. [Phase 3 — Regression & Classification MLPs](#phase-3--regression--classification-mlps)
7. [Phase 4 — Building MLPs with the Keras Sequential API](#phase-4--building-mlps-with-the-keras-sequential-api)
8. [Phase 5 — Functional API, Saving, Callbacks & Hyperparameter Tuning](#phase-5--functional-api-saving-callbacks--hyperparameter-tuning)
9. [Phase 6 — Unstable Gradients: Initialization, Activations, Batch Norm & Reusing Layers](#phase-6--unstable-gradients-initialization-activations-batch-norm--reusing-layers)
10. [Phase 7 — Faster Optimizers & Learning Rate Scheduling](#phase-7--faster-optimizers--learning-rate-scheduling)
11. [Phase 8 — Regularization & Capstone](#phase-8--regularization--capstone)
12. [Exercises (Must Do)](#exercises-must-do)
13. [Deliverables Checklist](#deliverables-checklist)
14. [Reference: Key Concepts Glossary](#reference-key-concepts-glossary)
15. [Resources](#resources)

---

## 1. Project Overview

### The Scenario

Every model in Tasks 7–10 was a classical algorithm with a closed-form or well-behaved optimizer. Neural networks are different: they are flexible enough to model almost anything, but they are also easy to train badly. A network with the wrong initialization, a saturating activation, or an untuned learning rate will simply sit there, or blow up. Task 16 is the practical manual for that: first the building blocks (neuron → Perceptron → MLP → backprop → Keras), then the training toolkit that turns a model that *could* work into one that *does*.

### Why This Matters

| Concept | Where It Shows Up in Practice |
|---|---|
| Perceptron / MLP / activations | The skeleton of every deep model, including the DNNs and towers in Tasks 14–15 |
| Backpropagation | Why `model.fit()` can train millions of parameters from a single loss |
| Keras Sequential / Functional API | Sequential for plain stacks; Functional for multi-input/multi-output models like Wide & Deep (Task 14) |
| Callbacks & early stopping | Checkpointing long runs and stopping before overfitting, for free |
| Initialization, ReLU variants, Batch Norm | The fixes for vanishing/exploding gradients that made *deep* learning possible |
| Optimizers & LR schedules | Often the difference between a model that converges in minutes and one that never does |
| Dropout / MC Dropout | The default regularizer for deep nets, and a cheap way to get uncertainty estimates |

---

## 2. What You Will Learn

By completing this task, you will be able to:

- Describe a threshold logic unit (TLU) and a Perceptron, and explain why a single-layer Perceptron can only learn linear decision boundaries (and cannot solve XOR)
- Write the Perceptron learning rule (`w_ij ← w_ij + η(y_j − ŷ_j)x_i`) and explain how it relates to Hebb's rule and to SGD
- Compute a dense layer's output as `ϕ(XW + b)` and state the shape of `X`, `W`, and `b`
- Explain the Multilayer Perceptron (input layer, hidden layers, output layer) and why a stack of purely linear layers collapses into one linear layer
- Walk through backpropagation in order: forward pass → loss → chain-rule backward pass → Gradient Descent step, and explain why weights must be initialized randomly (symmetry breaking)
- Compare sigmoid, tanh, and ReLU, and explain why ReLU became the default
- Choose output-layer activation and loss for regression (none / ReLU / softplus; MSE / MAE / Huber), binary, multilabel, and multiclass classification (sigmoid / softmax; cross-entropy)
- Build, compile, train, evaluate, and predict with a Keras Sequential model on Fashion MNIST, and read the training curves for over/underfitting
- Choose correctly between `sparse_categorical_crossentropy`, `categorical_crossentropy`, and `binary_crossentropy`
- Build a Wide & Deep model with the Functional API, and know when to use the Subclassing API instead
- Save/restore models, and use `ModelCheckpoint`, `EarlyStopping`, and TensorBoard callbacks
- Run a randomized hyperparameter search over a Keras model using the Scikit-Learn wrappers, and apply the guidelines for number of layers, neurons, learning rate, batch size, and optimizer
- Explain the vanishing/exploding gradients problem and apply Glorot, He, and LeCun initialization to the matching activation function
- Use leaky ReLU, ELU, and SELU to avoid "dying ReLUs"
- Write the four steps of Batch Normalization and add BN layers in Keras
- Apply gradient clipping (by value vs. by norm), and reuse pretrained layers via transfer learning
- Write the update rules for Momentum, Nesterov, AdaGrad, RMSProp, and Adam, and state which to avoid and why
- Implement power, exponential, piecewise-constant, performance, and 1cycle learning-rate schedules
- Apply ℓ1/ℓ2 regularization, dropout, MC Dropout, and max-norm regularization, and explain what each one prevents

---

## 3. Prerequisites & Setup

### Required Knowledge

- Task 7 (Gradient Descent, learning rate, softmax/logistic regression, early stopping, ℓ1/ℓ2 regularization) — reused throughout
- Task 1 (train/validation/test discipline, `RandomizedSearchCV`)
- Task 14 (Wide & Deep, softmax DNN) — helpful context, not required

### Environment Setup

```bash
pip install tensorflow scikit-learn matplotlib pandas --break-system-packages
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
```

> **API note:** older Keras code (including many tutorials) uses `lr=` for the learning rate and `np.int`. In current TensorFlow/Keras use `learning_rate=` and `int`. Where a Keras call below differs across versions, prefer the current spelling.

### Datasets

- **Fashion MNIST** (`keras.datasets.fashion_mnist`) — 70,000 grayscale 28×28 images, 10 clothing classes; harder than MNIST (a linear model gets ~83% vs. ~92%)
- **California Housing** — reuse your Task 1 data for the regression MLP

---

## Phase 1 — From Biological Neurons to the Perceptron



### Tasks to Perform

1. Summarize in a few sentences how ANNs were inspired by biological neurons, and why the biological analogy should not be pushed too far (planes were inspired by birds but don't flap).
2. Define a **threshold logic unit (TLU)**: it computes `z = xᵀw`, then applies a step function, `h_w(x) = step(z)`. Write both common step functions (Heaviside and sign).
3. Define a **Perceptron**: a single layer of TLUs, each connected to all inputs (a fully connected / dense layer), with passthrough input neurons and a bias neuron (`x0 = 1`).
4. Write the vectorized layer equation `h_{W,b}(X) = ϕ(XW + b)` and state the shape of `X` (instances × features), `W` (inputs × neurons), and `b` (one per neuron).
5. Write the **Perceptron learning rule**: `w_{i,j}(next) = w_{i,j} + η(y_j − ŷ_j)·x_i`, and explain how it was inspired by Hebb's rule ("cells that fire together wire together") but corrects for the error.
6. State the **Perceptron convergence theorem**: if the training data is linearly separable, the algorithm converges — but the solution is not unique, and Perceptrons output hard decisions, not probabilities.
7. Train `sklearn.linear_model.Perceptron` on iris (petal length/width, "is setosa?") and confirm its equivalence to `SGDClassifier(loss="perceptron", learning_rate="constant", eta0=1, penalty=None)`.
8. Explain, with the XOR problem, why a single Perceptron cannot learn it (XOR is not linearly separable), and why stacking Perceptrons (an MLP) solves it.

### 1.1 Formula Reference

```
TLU:                 z = xᵀw ;  h_w(x) = step(z)
Layer output:        h_{W,b}(X) = ϕ(XW + b)
Perceptron rule:     w_ij(next) = w_ij + η (y_j − ŷ_j) x_i
```

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron

iris = load_iris()
X = iris.data[:, (2, 3)]                 # petal length, petal width
y = (iris.target == 0).astype(int)       # Iris setosa?

per_clf = Perceptron()
per_clf.fit(X, y)
print(per_clf.predict([[2, 0.5]]))
```

**Deliverable for Phase 1:** Your written TLU/Perceptron summary with the learning rule, the trained `Perceptron`, and a hand-verified XOR argument (why one TLU fails, how two layers succeed).

---

## Phase 2 — The Multilayer Perceptron, Backpropagation & Activation Functions



### Tasks to Perform

1. Define an **MLP**: one input layer, one or more hidden layers of TLUs, one output layer; every layer except the output has a bias neuron and is fully connected to the next. Define *feedforward*, *lower/upper layers*, and when an ANN counts as a **deep neural network (DNN)**.
2. Explain **backpropagation** in your own words as Gradient Descent plus an efficient way to compute all gradients in two passes. Write the steps in order:
   - handle one mini-batch at a time; one pass over the training set is an **epoch**
   - **forward pass**: compute every layer's output, keeping intermediates
   - measure the output error with a loss function
   - **backward pass**: use the chain rule to find each connection's contribution to the error, layer by layer
   - take a Gradient Descent step using those gradients
3. Name the autodiff technique backprop uses (**reverse-mode autodiff**) and why it suits "many parameters, one loss."
4. Explain why all weights must be **initialized randomly**: with identical (e.g., zero) initialization every neuron in a layer stays identical and the layer behaves like a single neuron (symmetry).
5. Explain why the step function had to be replaced by the logistic function for backprop (flat segments ⇒ no gradient).
6. Write and sketch the three classic activations: **logistic** `σ(z) = 1/(1+e^{−z})`, **tanh** `tanh(z) = 2σ(2z) − 1` (range −1 to 1), and **ReLU** `max(0, z)`. State why ReLU is the default (fast, no saturation for positive values) and its caveat (not differentiable at 0, zero gradient for `z < 0`).
7. Prove, with `f(x) = 2x + 3` and `g(x) = 5x − 1`, that chaining linear functions gives another linear function (`f(g(x)) = 10x + 1`) — and conclude why nonlinear activations between layers are mandatory.
8. State the universal-approximation intuition: a large enough network with nonlinear activations can approximate any continuous function.

### 2.1 Reference

| Activation | Formula | Range | Note |
|---|---|---|---|
| Logistic (sigmoid) | `1/(1+e^{−z})` | (0, 1) | Saturates; mean 0.5 |
| tanh | `2σ(2z) − 1` | (−1, 1) | Centered at 0, often converges faster than sigmoid |
| ReLU | `max(0, z)` | [0, ∞) | Default hidden activation; can "die" |

**Deliverable for Phase 2:** A written, ordered walkthrough of backpropagation (5 steps), the symmetry-breaking explanation, and the "linear ∘ linear = linear" proof.

---

## Phase 3 — Regression & Classification MLPs



### Tasks to Perform

1. **Regression MLP:** one output neuron per predicted value (e.g., 1 for house price; 4 for a bounding box: x, y, width, height). Default output activation is **none**; use ReLU or **softplus** (`log(1+e^z)`) to force positive outputs, or logistic/tanh with rescaled labels to bound outputs.
2. Choose the regression loss: **MSE** by default, **MAE** if there are many outliers, **Huber** as the compromise (quadratic below threshold δ, linear above).
3. Fill in the regression-MLP architecture table (below) from memory, then compare with the reference.
4. **Binary classification:** one output neuron with logistic activation → probability of the positive class.
5. **Multilabel binary classification:** one logistic output neuron per label; probabilities need *not* sum to 1.
6. **Multiclass classification:** one output neuron per class with **softmax** (probabilities sum to 1; classes mutually exclusive). Loss is **cross-entropy** in all classification cases.
7. Complete the classification table from memory, then compare.
8. Do the TensorFlow Playground exercise (see Exercises) to see how layers, neurons, and activations change the decision boundary.

### 3.1 Architecture Tables

| Hyperparameter | Regression MLP |
|---|---|
| Input neurons | One per input feature |
| Hidden layers | Typically 1–5 |
| Neurons per hidden layer | Typically 10–100 |
| Output neurons | 1 per prediction dimension |
| Hidden activation | ReLU (or SELU) |
| Output activation | None; ReLU/softplus (positive); logistic/tanh (bounded) |
| Loss | MSE, or MAE/Huber with outliers |

| Hyperparameter | Binary | Multilabel binary | Multiclass |
|---|---|---|---|
| Input & hidden layers | same as regression | same | same |
| Output neurons | 1 | 1 per label | 1 per class |
| Output activation | Logistic | Logistic | Softmax |
| Loss | Cross-entropy | Cross-entropy | Cross-entropy |

**Deliverable for Phase 3:** Both tables completed from memory, plus a recommendation (output activation, loss, #outputs) for four scenarios: house-price prediction, spam vs. ham, spam + urgent tagging, 10-digit recognition.

---

## Phase 4 — Building MLPs with the Keras Sequential API



### Tasks to Perform

1. Load Fashion MNIST, create a **validation set** from the first 5,000 training images, and **scale pixels to 0–1** by dividing by 255 (inputs must be scaled for Gradient Descent).
2. Define `class_names` and confirm what the first training image is.
3. Build a Sequential classifier: `Flatten(input_shape=[28, 28])` → `Dense(300, relu)` → `Dense(100, relu)` → `Dense(10, softmax)`.
4. Call `model.summary()`; verify the parameter count by hand for each Dense layer (`inputs × neurons + neurons`).
5. **Compile** with `loss="sparse_categorical_crossentropy"`, `optimizer="sgd"`, `metrics=["accuracy"]`. Explain *why* sparse (integer class labels) vs. `categorical_crossentropy` (one-hot) vs. `binary_crossentropy` (sigmoid outputs). Use `keras.utils.to_categorical()` / `np.argmax(..., axis=1)` to convert between label formats.
6. Note that `optimizer="sgd"` uses `learning_rate=0.01`; set it explicitly with `keras.optimizers.SGD(learning_rate=...)` when tuning.
7. **Train** for 30 epochs with `validation_data=(X_valid, y_valid)`. Plot `history.history` (train/val loss and accuracy per epoch).
8. Diagnose from the curves: validation much worse than training ⇒ overfitting (or a train/validation data mismatch bug); both still improving ⇒ keep training.
9. Evaluate on the test set with `model.evaluate`, then use `predict` / `argmax` on a few images and compare with true labels. Remember: **only evaluate on the test set once, after tuning**.
10. Build a **regression MLP** on California Housing with the Sequential API (`Dense(30, relu)` → `Dense(1)`), compile with `loss="mean_squared_error"`, and compare its test MSE to your best Task 1 model. Remember to scale features (e.g., `StandardScaler` fit on train only).

### 4.1 Code Reference

```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=0.01),
              metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
pd.DataFrame(history.history).plot(figsize=(8, 5)); plt.grid(True); plt.show()
```

**Deliverable for Phase 4:** The trained Fashion MNIST classifier with its learning-curve plot, test-set accuracy, a diagnosis paragraph, and the California Housing regression MLP's test MSE next to your Task 1 baseline.

---

## Phase 5 — Functional API, Saving, Callbacks & Hyperparameter Tuning



### Tasks to Perform

1. Explain when the Sequential API isn't enough: non-sequential topologies, multiple inputs, multiple outputs.
2. Build a **Wide & Deep** regression model with the Functional API: `Input` → two `Dense(30, relu)` layers → `Concatenate()([input_, hidden2])` → `Dense(1)`. Explain what the short ("wide") path adds (simple rules + manually engineered features) and connect it to the Wide & Deep architecture from Task 14, Phase 7.
3. Extend it to **multiple inputs** (e.g., one subset of features through the wide path, another through the deep path) and **multiple outputs** (a main output plus an auxiliary output used as a regularizer). Compile with a list of losses and `loss_weights`.
4. Explain the **Subclassing API** (define layers in `__init__`, the computation in `call()`): maximal flexibility for dynamic models, but the model is opaque to Keras (no `summary()` structure, can't `save()` the whole model, only `save_weights()`).
5. **Save and restore** a model with `model.save("my_keras_model.h5")` and `keras.models.load_model(...)`; note this works for Sequential/Functional but not Subclassing.
6. Use callbacks:
   - `ModelCheckpoint(..., save_best_only=True)` → roll back to the best validation model
   - `EarlyStopping(patience=10, restore_best_weights=True)` → stop when validation stalls
   - Combine them, and add a custom callback that prints the validation-to-training loss ratio each epoch to catch overfitting early.
7. Use **TensorBoard**: log to a timestamped directory (one run per subdirectory), launch `tensorboard --logdir`, and compare learning curves across runs.
8. **Fine-tune hyperparameters:** wrap a `build_model(n_hidden, n_neurons, learning_rate, input_shape)` function with `keras.wrappers.scikit_learn.KerasRegressor` (or the modern `scikeras` equivalent), then run `RandomizedSearchCV` over `n_hidden`, `n_neurons`, and `learning_rate` (use `scipy.stats.reciprocal(3e-4, 3e-2)`). Remember `RandomizedSearchCV` uses K-fold CV, so the validation set only serves early stopping.
9. Name libraries that explore the space more efficiently than random search (Hyperopt, Keras Tuner, Scikit-Optimize, Spearmint, Hyperband, Hyperas/kopt/Talos, Population-Based Training) and the core idea: explore promising regions more.
10. Record the book's **tuning guidelines**:
    - *Hidden layers:* start with 1; deep nets are far more parameter-efficient than shallow ones for complex problems (hierarchical features); add layers until you start overfitting.
    - *Neurons per layer:* the same count in all hidden layers works as well as a pyramid; prefer "stretch pants" — pick slightly too many layers/neurons, then use early stopping and regularization.
    - *Learning rate:* the most important hyperparameter; optimal ≈ half the maximum rate that doesn't diverge. Find it by training a few hundred iterations while growing the rate exponentially (e.g., 10⁻⁵ → 10) and picking a value ~10× below where the loss shoots up.
    - *Batch size:* large is faster on GPU but can be unstable/generalize worse; strategy — try large with learning-rate warmup, fall back to small (≤32) if unstable.
    - *Activation:* ReLU default for hidden layers; output depends on the task.

### 5.1 Code Reference

```python
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])
```

```python
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
```

**Deliverable for Phase 5:** The Wide & Deep model (single- and multi-input versions), a saved and reloaded model, a run with both callbacks, a TensorBoard comparison of ≥2 runs, and the `RandomizedSearchCV` result with best parameters.

---

## Phase 6 — Unstable Gradients: Initialization, Activations, Batch Norm & Reusing Layers



### Tasks to Perform

1. Define the **vanishing gradients** problem (gradients shrink going down to lower layers, so lower layers barely train) and the **exploding gradients** problem (gradients grow, weights diverge — common in recurrent nets). Explain why the logistic activation plus the old normal(0, 1) initialization made this worse: layer-output variance grows with depth until the function saturates and its derivative is ≈ 0.
2. State the Glorot–Bengio insight: signal must flow properly in both directions, so the variance of each layer's outputs should equal that of its inputs (and likewise for gradients going backward).
3. Fill in the initialization table and match each to its activation. Keras defaults to Glorot uniform; switch with `kernel_initializer="he_normal"` (or `"lecun_normal"` for SELU).
4. Explain the **dying ReLU** problem and the fixes: **leaky ReLU** (`max(αz, z)`, α ≈ 0.01–0.2), **RReLU**, **PReLU** (learned α), and **ELU** (`α(e^z − 1)` for `z < 0`, `z` otherwise: negative values pull the mean toward 0, smooth everywhere).
5. Explain **SELU** (scaled ELU): a deep stack of Dense layers using SELU with LeCun-normal init and standardized inputs can *self-normalize*; state its conditions (plain sequential MLP, standardized inputs, LeCun normal init, no regularization that breaks it such as plain dropout — use AlphaDropout).
6. Rule of thumb (from the book): SELU > ELU > leaky ReLU (and variants) > ReLU > tanh > logistic, trading off speed vs. performance.
7. Write the four **Batch Normalization** steps per mini-batch: mean `μ_B`, variance `σ²_B`, normalize `x̂ = (x − μ_B)/√(σ²_B + ε)`, then scale and shift `z = γ ⊗ x̂ + β` (γ and β are learned). Explain train vs. test behavior (test time uses moving averages of μ and σ computed during training).
8. Add BN to your Fashion MNIST model (after each Dense, before or after the activation — try both) and compare convergence speed and accuracy to Phase 4. Note BN also acts as a regularizer, and can remove the need for a `StandardScaler` if used as the first layer.
9. Explain **gradient clipping** (`clipvalue=1.0` clips each component and can change direction; `clipnorm=1.0` rescales the whole vector and preserves direction) and when it's used (mostly recurrent nets, where BN is tricky).
10. Explain **transfer learning**: reuse the lower layers of a model trained on a similar task. Build it in Keras: clone model A's layers (`keras.models.clone_model` + copy weights), freeze reused layers (`layer.trainable = False`, recompile), train the new top, then unfreeze the top few layers and fine-tune with a *lower* learning rate.
11. Describe **unsupervised pretraining** (autoencoders/GANs, then fine-tune with labels) and **pretraining on an auxiliary task** (e.g., train on a task where labels are cheap, then reuse the layers) as options when labeled data is scarce.

### 6.1 Reference

| Initialization | Activation | σ² (Normal) |
|---|---|---|
| Glorot | None, tanh, logistic, softmax | 1 / fan_avg |
| He | ReLU and variants | 2 / fan_in |
| LeCun | SELU | 1 / fan_in |

```
Batch Norm:  μ_B  = (1/m_B) Σ x⁽ⁱ⁾
             σ²_B = (1/m_B) Σ (x⁽ⁱ⁾ − μ_B)²
             x̂⁽ⁱ⁾ = (x⁽ⁱ⁾ − μ_B) / √(σ²_B + ε)
             z⁽ⁱ⁾  = γ ⊗ x̂⁽ⁱ⁾ + β
```

```python
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(100, kernel_initializer="he_normal", use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10, activation="softmax"),
])
```

```python
# Transfer learning sketch
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                     metrics=["accuracy"])
```

**Deliverable for Phase 6:** A comparison table of Phase 4's baseline vs. He-init+ELU vs. +BatchNorm (epochs to a target accuracy, final validation accuracy), a gradient-clipping explanation with the `[0.9, 100.0]` example worked by hand (value → `[0.9, 1.0]`; norm → `[0.00899964, 0.9999595]`), and a working transfer-learning experiment.

---

## Phase 7 — Faster Optimizers & Learning Rate Scheduling



### Tasks to Perform

1. Write plain Gradient Descent: `θ ← θ − η∇θJ(θ)`, and explain its weakness (ignores past gradients; slow on plateaus and in elongated valleys).
2. **Momentum:** `m ← βm − η∇θJ(θ)`; `θ ← θ + m`. Interpret β as friction (typically 0.9); show that terminal velocity is `η·∇/(1−β)` (10× faster at β = 0.9). In Keras: `SGD(learning_rate=0.001, momentum=0.9)`.
3. **Nesterov Accelerated Gradient (NAG):** measure the gradient slightly *ahead* in the momentum direction, `∇θJ(θ + βm)`. In Keras: `SGD(..., momentum=0.9, nesterov=True)`; it usually beats plain momentum.
4. **AdaGrad:** `s ← s + ∇⊗∇`; `θ ← θ − η ∇ ⊘ √(s + ε)`. Explain the adaptive learning rate (steep dimensions decay faster) and why you should **not** use it for deep networks (the rate decays so much it stops early).
5. **RMSProp:** `s ← βs + (1−β)∇⊗∇` (exponential decay, β ≈ 0.9), then the same update. Explain how it fixes AdaGrad.
6. **Adam:** momentum + RMSProp with bias correction:
   `m ← β₁m − (1−β₁)∇`; `s ← β₂s + (1−β₂)∇⊗∇`; `m̂ ← m/(1−β₁ᵗ)`; `ŝ ← s/(1−β₂ᵗ)`; `θ ← θ + η m̂ ⊘ √(ŝ + ε)`. Defaults: `β₁=0.9`, `β₂=0.999`, `ε=1e-7`. Mention AdaMax and Nadam as variants.
7. Train the Phase 6 model with SGD, Momentum, Nesterov, RMSProp, and Adam and **plot validation loss vs. epoch for all five** on one chart.
8. Explain why adaptive optimizers sometimes generalize worse than plain (Nesterov) momentum — if Adam disappoints on your problem, try Nesterov.
9. Implement the **learning-rate schedules**:
   - **Power:** `η(t) = η₀/(1 + t/s)^c` (drops to η₀/2 after s steps, η₀/3 after 2s, …)
   - **Exponential:** `η(t) = η₀·0.1^{t/s}` (÷10 every s steps)
   - **Piecewise constant:** e.g., 0.1 for 5 epochs, then 0.001 for 50 epochs
   - **Performance:** `ReduceLROnPlateau(factor=0.5, patience=5)`
   - **1cycle:** rise linearly from η₀ to η₁ (≈10× higher) over the first half, fall back over the second half, then drop several orders of magnitude at the end; momentum moves in the opposite direction (e.g., 0.95 → 0.85 → 0.95)
10. Use `keras.callbacks.LearningRateScheduler` (or `keras.optimizers.schedules`) for at least two schedules and compare against a constant rate.
11. Find the best learning rate with the exponential-growth sweep from Phase 5 and use it as η₁ for 1cycle.

### 7.1 Formula Reference

```
Momentum:   m ← βm − η∇θJ(θ) ;           θ ← θ + m
AdaGrad:    s ← s + ∇⊗∇ ;                θ ← θ − η ∇ ⊘ √(s+ε)
RMSProp:    s ← βs + (1−β)∇⊗∇ ;          θ ← θ − η ∇ ⊘ √(s+ε)
Adam:       m ← β₁m − (1−β₁)∇ ;  s ← β₂s + (1−β₂)∇⊗∇
            m̂ ← m/(1−β₁ᵗ) ;      ŝ ← s/(1−β₂ᵗ)
            θ ← θ + η m̂ ⊘ √(ŝ+ε)
```

```python
optimizers = {
    "sgd": keras.optimizers.SGD(learning_rate=0.01),
    "momentum": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    "nesterov": keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
    "rmsprop": keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9),
    "adam": keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
}
```

```python
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
```

**Deliverable for Phase 7:** The five-optimizer comparison plot, a learning-rate-schedule comparison (constant vs. ≥2 schedules), the learning-rate-finder plot, and a one-paragraph recommendation of which optimizer + schedule you'd use by default.

---

## Phase 8 — Regularization & Capstone



### Tasks to Perform

1. List the regularizers already in your toolkit: **early stopping** (Phase 5) and **Batch Normalization** (Phase 6), plus the new ones below.
2. Apply **ℓ2** (`kernel_regularizer=keras.regularizers.l2(0.01)`), **ℓ1** (sparse weights), and **ℓ1+ℓ2** (`l1_l2`) to Dense layers. Use `functools.partial` to build a `RegularizedDense` wrapper with default activation/initializer/regularizer so you don't repeat arguments.
3. **Dropout:** at each training step every neuron (except output neurons) is dropped with probability `p` and ignored for that step. Typical `p`: 10–50% (≈20–30% in recurrent nets, ≈40–50% in convolutional nets). After training nothing is dropped. Explain the scaling detail: remaining activations are scaled so expected totals match between train and test. Add `keras.layers.Dropout(rate=0.2)` before every Dense layer and compare train vs. validation accuracy gaps with and without it. Remember: if the model **underfits**, lower the rate; if it **overfits**, raise it.
4. Note that with SELU you should use **AlphaDropout**.
5. **Monte Carlo (MC) Dropout:** keep dropout active at inference and average many stochastic predictions. Implement:
   `y_probas = np.stack([model(X_test_scaled, training=True) for _ in range(100)])`; `y_proba = y_probas.mean(axis=0)`; `y_std = y_probas.std(axis=0)`.
   Reproduce the book's observation: a single dropout-off prediction can claim 99% confidence while the MC-averaged prediction shows ~62% with visible alternatives — a far more honest uncertainty estimate, and often a small accuracy boost, at the cost of ~N× inference time.
6. If your model contains layers that behave differently in training (e.g., `BatchNormalization`), do **not** force `training=True` globally; instead subclass: `class MCDropout(keras.layers.Dropout): def call(self, inputs): return super().call(inputs, training=True)`.
7. **Max-norm regularization:** constrain each neuron's incoming weight vector so `‖w‖₂ ≤ r`; implemented by rescaling `w ← w·r/‖w‖₂` after each step (in Keras: `kernel_constraint=keras.constraints.max_norm(1.)`).
8. **Capstone:** combine everything into one strong model on Fashion MNIST (or a dataset of your choice): He init + ELU (or SELU for a pure-MLP variant) → Batch Norm → Dropout → Nadam/Adam (or Nesterov SGD) with 1cycle or `ReduceLROnPlateau` → `EarlyStopping` + `ModelCheckpoint` + TensorBoard. Report validation/test accuracy vs. the Phase 4 baseline and an ablation table showing what each ingredient contributed.
9. Add MC Dropout to the capstone and flag the 10 least-confident test predictions by `y_std`; discuss how you'd route them to a human reviewer in a risk-sensitive system.
10. Write the final guidance table below in your own words, using your experiments to justify each row.

### 8.1 Default-Configuration Guideline (verify against your experiments)

| Hyperparameter | Default (plain deep net) |
|---|---|
| Kernel initializer | He |
| Activation | ELU |
| Normalization | Batch Normalization (shallower nets: may be unnecessary) |
| Regularization | Early stopping (+ ℓ2 if needed) |
| Optimizer | Momentum/Nesterov, RMSProp, or Adam/Nadam |
| Learning rate schedule | 1cycle or performance scheduling |
| Self-normalizing alternative | LeCun init + SELU, standardized inputs, no BN, AlphaDropout if needed |

**Deliverable for Phase 8:** The regularization comparison (none vs. ℓ2 vs. dropout vs. max-norm), the MC Dropout uncertainty demo, the capstone model with its ablation table, and your completed default-configuration table.

---

## Exercises (Must Do)

| # | Exercise | Skills Practiced |
|---|----------|--------------------|
| 1 | In the TensorFlow Playground, solve the spiral dataset with ≤ 8 neurons and ≤ 3 hidden layers; then see which activation and which added features help most | MLP intuition |
| 2 | Draw an ANN from the original artificial-neuron formulation computing `A ⊕ B` (XOR) using TLUs; verify its truth table by hand | Logical computation with neurons |
| 3 | Why is logistic regression-style `sigmoid` output right for binary classification but `softmax` right for exclusive multiclass? What changes for multilabel? | Output layer design |
| 4 | Train the Fashion MNIST MLP with a learning rate that is too high and too low; describe the learning curves; then use the exponential LR sweep to find a good value | LR diagnosis |
| 5 | Build a deep MLP (20 hidden layers × 100 neurons, Nadam, He init) on CIFAR10-flattened or Fashion MNIST; compare with and without Batch Normalization, and with SELU + standardized inputs | Unstable gradients, normalization |
| 6 | Plot the sampled values of an untrained dense layer's pre-activations with `Normal(0,1)` init vs. Glorot vs. He across 10 layers deep; observe variance growth | Initialization experiment |
| 7 | Add dropout to your best model; evaluate with MC Dropout (N = 10, 50, 100) and plot accuracy vs. N and inference time vs. N | MC Dropout trade-offs |
| 8 | Transfer learning: train a small classifier on 8 Fashion MNIST classes, reuse its lower layers for the remaining 2, and compare to training from scratch with only 200 labeled examples | Transfer learning |
| 9 | Run Keras Tuner (or `RandomizedSearchCV`) over number of layers, neurons, learning rate, and dropout rate; report the best config and the compute budget | Hyperparameter search |
| 10 | Pair-writing: explain to a teammate, in ≤ 150 words each, (a) why backprop needs a forward pass first, (b) why BN has different train/test behavior | Teaching-level understanding |

---

## Deliverables Checklist

- [ ] **Phase 1:** TLU/Perceptron summary + learning rule + trained `Perceptron` + XOR argument
- [ ] **Phase 2:** 5-step backprop walkthrough + symmetry-breaking explanation + linear-collapse proof
- [ ] **Phase 3:** Both architecture tables from memory + 4-scenario output/loss recommendations
- [ ] **Phase 4:** Fashion MNIST classifier with learning curves + test accuracy + housing regression MLP vs. Task 1 baseline
- [ ] **Phase 5:** Wide & Deep model, saved/reloaded model, callbacks run, TensorBoard comparison, randomized search result
- [ ] **Phase 6:** Initialization/activation/BN comparison table + gradient-clipping worked example + transfer-learning experiment
- [ ] **Phase 7:** Five-optimizer comparison plot + LR-schedule comparison + LR-finder plot + recommendation
- [ ] **Phase 8:** Regularization comparison + MC Dropout demo + capstone with ablation table + default-configuration table
- [ ] **Exercises:** All 10 exercises attempted
- [ ] **Code:** Clean notebook that runs top to bottom without errors

---

## Reference: Key Concepts Glossary

| Term | Definition |
|------|------------|
| **Threshold Logic Unit (TLU)** | An artificial neuron that computes a weighted sum of inputs and applies a step function |
| **Perceptron** | A single layer of TLUs; learns linear decision boundaries via the Perceptron rule |
| **Perceptron Learning Rule** | `w_ij ← w_ij + η(y_j − ŷ_j)x_i`; reinforces connections that reduce error |
| **Hebb's Rule** | "Cells that fire together wire together"; strengthens connections between co-active neurons |
| **Multilayer Perceptron (MLP)** | Input layer + ≥1 hidden layers + output layer; feedforward and fully connected |
| **Deep Neural Network (DNN)** | An ANN with a deep stack of hidden layers |
| **Backpropagation** | Forward pass, loss, reverse-mode autodiff backward pass, then Gradient Descent step |
| **Epoch** | One pass over the full training set |
| **Reverse-Mode Autodiff** | Computes gradients of one output w.r.t. many parameters efficiently; used by backprop |
| **Activation Function** | Nonlinearity applied after a layer's linear transformation (sigmoid, tanh, ReLU, ELU, SELU…) |
| **Softplus** | `log(1 + e^z)`; smooth ReLU, used to force positive regression outputs |
| **Huber Loss** | Quadratic for small errors, linear for large ones; robust to outliers |
| **Sequential / Functional / Subclassing API** | Three Keras ways to define models, in increasing order of flexibility |
| **Callback** | An object Keras calls at training events (e.g., `ModelCheckpoint`, `EarlyStopping`, TensorBoard) |
| **Vanishing / Exploding Gradients** | Gradients shrinking or growing as they propagate through layers, preventing stable training |
| **Glorot / He / LeCun Initialization** | Variance-scaled random initializers matched to activation functions |
| **Dying ReLU** | A neuron stuck outputting 0 for all inputs, receiving no gradient |
| **Leaky ReLU / PReLU / ELU / SELU** | ReLU variants that avoid dying units; SELU can self-normalize deep MLPs |
| **Batch Normalization** | Normalizes layer inputs per mini-batch, then scales/shifts with learned γ and β |
| **Gradient Clipping** | Capping gradients by value or by norm to prevent explosions |
| **Transfer Learning** | Reusing lower layers of a pretrained network for a related task |
| **Momentum / Nesterov** | Optimizers that accumulate past gradients as velocity (Nesterov evaluates the gradient ahead) |
| **AdaGrad / RMSProp / Adam / Nadam** | Adaptive learning-rate optimizers; Adam = momentum + RMSProp, Nadam = Adam + Nesterov |
| **Learning Schedule** | A rule for changing the learning rate over training (power, exponential, piecewise, performance, 1cycle) |
| **Dropout** | Randomly ignoring neurons during each training step with probability `p` to reduce overfitting |
| **MC Dropout** | Keeping dropout active at inference and averaging many predictions for better accuracy and uncertainty estimates |
| **Max-Norm Regularization** | Constraining each neuron's incoming weight norm to ≤ r |

---

## Resources

| Resource | Link |
|----------|------|
| Keras documentation | [keras.io](https://keras.io/) |
| TensorFlow Playground | [playground.tensorflow.org](https://playground.tensorflow.org/) |
| Keras Tuner | [keras.io/keras_tuner](https://keras.io/keras_tuner/) |
| TensorBoard guide | [tensorflow.org/tensorboard](https://www.tensorflow.org/tensorboard) |
| Glorot & Bengio (2010), "Understanding the Difficulty of Training Deep Feedforward Neural Networks" | Search via Google Scholar |
| Ioffe & Szegedy (2015), "Batch Normalization" | [arxiv.org/abs/1502.03167](https://arxiv.org/abs/1502.03167) |
| Kingma & Ba (2014), "Adam: A Method for Stochastic Optimization" | [arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980) |
| Gal & Ghahramani (2016), "Dropout as a Bayesian Approximation" | [arxiv.org/abs/1506.02142](https://arxiv.org/abs/1506.02142) |
| Task 7 reference document | `Task_7_Training_Models.md` |
| Task 14 reference document | `Task_14_Deep_Learning_for_Recommendations.md` |

---

## Suggested Timeline

| Date | Phases |
|------|--------|
| 15/10/2026| Phase 1–8, Exercises |

---
**NOTE:** As in previous tasks, attend daily standups/wrap-ups and keep the tracking sheet updated.