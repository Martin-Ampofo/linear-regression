# Linear Regression from Scratch

This repository contains a clean, modular implementation of **Linear Regression** built as a machine learning assignment. Instead of relying on high-level libraries like Scikit-Learn for the model, this implementation is built entirely from the ground up using foundational mathematical equations.

## 🧮 Mathematical Foundations & Implementations

This project explores multiple analytical and optimization-based approaches to solving linear regression:

### 1. Normal Equation (Closed-Form Solution)
Calculates the optimal coefficients directly without iterative optimization steps:
θ = (XᵀX)⁻¹ Xᵀy

### 2. Locally Weighted Regression (LWR)
A non-parametric algorithm that makes predictions by weighting training points based on their distance to the query point $x$:
θ_q = (XᵀWX)⁻¹ XᵀWy
Where the weights $w^{(i)}$ are typically calculated using an exponential kernel:
w(i) = exp( −‖x(i) − x_q‖² / (2τ²) )

### 3. Batch Gradient Descent (BGD)
Iteratively minimizes the Cost Function $J(\theta)$ by calculating the gradient using the entire dataset at each step:
θ:=θ-α⋅1/m⋅X^T (Xθ-y)

### 4. Stochastic Gradient Descent (SGD)
Optimizes the parameters iteratively by updating weights using only a single, randomly selected training example at each step, significantly speeding up computation on larger datasets.
θ:=θ-α⋅1/m⋅X^T (Xθ-y)

---

## 📁 Repository Structure
* `linearRegression.ipynb`: The primary Jupyter Notebook containing data preprocessing, implementations of all four algorithms, model training, and visualizations.
* `.gitignore`: Configured to exclude local Jupyter checkpoint files (`.ipynb_checkpoints/`) to keep the repository clean.
