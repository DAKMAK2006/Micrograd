# 🧠 micrograd from Scratch: A Scalar Autograd Engine

## Project Description

This project implements a complete **scalar-valued automatic differentiation (autograd) engine** and a minimal Neural Network library from scratch in Python.

The core goal is to build the mathematical foundation of deep learning—specifically, the **reverse-mode backpropagation** algorithm—without relying on high-level libraries (like PyTorch or TensorFlow) for gradient computation. This implementation provides **full algorithmic transparency** into how the chain rule is applied during neural network training.

---

## 🛠️ Implementation Details

### 1. Automatic Differentiation Core

The project is driven by a custom `Value` object that forms a dynamically constructed computational graph:

* **`Value` Class:** A custom scalar data structure that holds both the numeric `data` and the computed `grad` (derivative) at every node.
* **Dynamic DAG:** Mathematical operations (`+`, `*`, `tanh`, `relu`, `pow`) are overloaded to dynamically build a **Directed Acyclic Graph (DAG)**, which tracks dependencies for the backward pass.
* **Backpropagation:** The `.backward()` method performs **reverse-mode backpropagation**, recursively applying the chain rule to compute and accumulate the gradient for all operations in the graph.

### 2. Neural Network Library

A small, functional Deep Learning library is built directly on top of the custom `Value` object:

* **`Neuron`, `Layer`, and `MLP`:** Implemented custom classes to handle the standard components of an **MLP (Multi-Layer Perceptron)**.
* **Activation Functions:** Supports non-linearities like $\mathbf{tanh}$ and $\mathbf{relu}$.

---

## 🚀 Demo & Results

The custom engine's capabilities are validated by training a complete classifier:

* **Dataset:** Trained a 2-layer MLP on a custom dataset for binary classification.
* **Loss & Optimization:** Used a custom **max-margin binary classification loss** (similar to SVM loss) and an iterative **Stochastic Gradient Descent (SGD)** loop for weight updates, demonstrating manual control over the optimization process.
* **Visualization:** The notebook includes visualization of the final **decision boundary** learned by the network, proving the autograd engine's functionality.

---

## 💡 Usage

1.  Clone this repository.
2.  Open the notebook: `micrograd_from_scratch.ipynb`.
3.  Execute the cells to follow the implementation of the `Value` class, build the MLP, and train the final classifier.
