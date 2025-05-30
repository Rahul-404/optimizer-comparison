# ⚙️ Optimizers in Deep Learning 🚀

---

## Overview 🔍

Optimizers play a crucial role in increasing the performance of neural networks by updating weights and activation functions during training.

The fundamental update rule for weights is:

$$
w_{new} = w_{old} - \eta \cdot \frac{\partial L}{\partial w_{old}}
$$

where  
- \( \eta \) is the learning rate 🎯,  
- \( \frac{\partial L}{\partial w_{old}} \) is the gradient of the loss with respect to the old weight.

---

## Types of Optimizers 🛠️

### 1. Gradient Descent Variants 🔄

Gradient Descent optimizers update weights to minimize loss. There are three main types:

- **Batch Gradient Descent (BGD) 🧮**
- **Stochastic Gradient Descent (SGD) 🎲**
- **Mini-Batch Gradient Descent 📦**

---

### Epochs and Iterations ⏳

- **Epoch:** One full pass over the entire dataset (big cycle) 🔄.
- **Iteration:** One update step of weights using a batch of data (small cycle) 🔁.

---

### 1. Batch Gradient Descent (BGD) 🧮

- Processes all data points at once to update weights.
- Example: For 1000 data points, 1 epoch = 1 iteration = 1000 data points processed before weight update.
- **Pros:** ✅ Converges steadily.
- **Cons:** ❌  
  - Memory intensive (OOM errors if dataset is huge).  
  - Slow and requires large computational resources.

*Example:*  
100 epochs × 1000 data points → 100,000 total weight updates.

**Problem:** What if dataset has 1 billion points? 🧠💥  
- Requires huge RAM, often causing out-of-memory (OOM) errors.

---

### 2. Stochastic Gradient Descent (SGD) 🎲

- Updates weights after processing **one data point** per iteration.
- Example: For 1000 data points, each epoch has 1000 iterations.
- **Pros:**  
  - No memory overflow issues 🧠❌.  
- **Cons:**  
  - Noisy updates, oscillations in convergence 🎢.  
  - Time-consuming ⏰.

---

### 3. Mini-Batch Gradient Descent 📦

- Splits data into batches smaller than the full dataset.
- Batch size is a hyperparameter ⚙️.
- Example:  
  - 1000 data points, batch size = 100 → 10 iterations per epoch.  
  - If batch size = 90, 11 iterations with last one processing remaining 10 points.

**Pros:**

- Reduced noise compared to SGD 🔕.
- Efficient memory usage 💾.
- Faster than pure SGD ⚡.

**Cons:**

- Still some noise in updates 🔊.

---

### Choosing Batch Size 🎚️

- Depends on GPU VRAM (common sizes are powers of 2: 2, 4, 8, 16, 32 GB, etc.).
- GPUs optimized for batch sizes of \(2^n\) due to hardware design (Nvidia, AMD) 🎮⚙️.

---

## Advanced Optimizers 🚀

### 2. SGD with Momentum 🏃‍♂️💨

Momentum helps reduce oscillations and speeds up convergence by considering past gradients.

Update formula considering time step \(t\):

$$
w_t = w_{t-1} - \eta \cdot \frac{\partial L}{\partial w_{t-1}}
$$

Momentum smooths the updates by giving "push" to escape plateaus in loss landscapes ⛰️.

**Benefits:**

- Noise reduction 🔇
- Smoother convergence 🛤️

---

### 3. Adagrad (Adaptive Gradient Descent) 📈

- Adapts learning rate dynamically based on past gradients.
- Learning rate update:

$$
\eta' = \frac{\eta}{\sqrt{\alpha^t + \epsilon}}
$$

- Where

$$
\alpha^t = \sum_{i=1}^t \left(\frac{\partial L}{\partial w_i}\right)^2
$$

- \( \epsilon \) is a small constant to avoid division by zero.

**Pros:** Larger steps at the start and smaller steps near convergence 🎯.

**Cons:** Slow convergence towards the end 🐢.

---

### 4. RMS Prop (Root Mean Square Propagation) 🌊

Improves Adagrad by using Exponentially Weighted Average (EWA) to prevent learning rate decay becoming too aggressive.

Update:

$$
\eta' = \frac{\eta}{\sqrt{Sd_{wt} + \epsilon}}
$$

Where:

$$
Sd_{wt} = \beta \cdot Sd_{wt-1} + (1 - \beta) \cdot \left(\frac{\partial L}{\partial w_t}\right)^2
$$

**Comparison:**

- RMSProp converges faster ⚡ and is less jittery 🎯 than Adagrad.

---

### 5. Adam Optimizer (Adaptive Moment Estimation) 🤖✨

Combines RMS Prop and SGD with Momentum.

Update rules:

$$
w_t = w_{t-1} + \eta' \cdot V_{dw_t}
$$

Where:

$$
V_{dw_t} = \beta \cdot V_{dw_{t-1}} + (1-\beta) \cdot \frac{\partial L}{\partial w_t}
$$

and

$$
\eta' = \frac{\eta}{\sqrt{Sd_{wt} + \epsilon}}
$$

Adam adjusts learning rate dynamically and includes momentum, making it a popular choice in industry 🏭.

---

## Learning Rate Scheduling ⏱️

Example Python code for manual scheduling of learning rate:

```python
epochs = 100

if epochs <= 25:
    lr = 0.01
elif 25 < epochs <= 30:
    lr = 0.001
elif 30 < epochs <= 75:
    lr = 0.0001
else:
    lr = 0.00001
```

This reference to learning rate scheduler

### **References 📚:**

- [https://www.ruder.io/tag/optimization/](https://www.ruder.io/tag/optimization/)

    - Optimization for Deep Learning Highlights in 2017
    - An overview of gradient descent optimization algorithms