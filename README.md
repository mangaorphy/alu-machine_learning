# Mathematics for Machine Learning

This repository contains implementations of fundamental machine learning algorithms and mathematical concepts, focusing on neural networks, classification, and supervised learning.

## 📚 Repository Structure

```
alu-machine_learning/
├── data/                           # Training datasets
│   ├── Binary_Dev.npz             # Binary classification development set
│   ├── Binary_Train.npz           # Binary classification training set
│   └── MNIST.npz                  # MNIST handwritten digits dataset
├── math/                          # Mathematical foundations
│   ├── advanced_linear_algebra/   # Matrix operations, determinants, eigenvalues
│   ├── bayesian_prob/            # Bayesian probability and inference
│   ├── calculus/                 # Derivatives, integrals, optimization
│   ├── convolutions_and_pooling/ # CNN operations
│   ├── linear_algebra/           # Vectors, matrices, transformations
│   ├── multivariate_prob/        # Multivariate statistics
│   ├── plotting/                 # Data visualization
│   └── probability/              # Probability distributions
└── supervised_learning/
    └── classification/           # Neural network implementations
```

## 🧠 Supervised Learning - Classification

The classification module contains a complete implementation of neural networks from scratch, progressing from simple neurons to deep neural networks with advanced features.

### Core Components

#### 1. Neuron Class (`0-neuron.py` to `7-neuron.py`)
- **Single Neuron Implementation**: Basic building block of neural networks
- **Features**:
  - Forward propagation with sigmoid activation
  - Cost calculation using logistic regression
  - Model evaluation with binary predictions
  - Gradient descent optimization
  - Training with monitoring and visualization
  
```python
# Example usage
neuron = Neuron(784)  # 784 input features (e.g., 28x28 MNIST images)
A, cost = neuron.train(X_train, Y_train, iterations=5000)
predictions, cost = neuron.evaluate(X_test, Y_test)
```

#### 2. Neural Network Class (`8-neural_network.py` to `15-neural_network.py`)
- **Single Hidden Layer Network**: Extension to multi-node hidden layer
- **Features**:
  - Configurable hidden layer size
  - Forward propagation through hidden and output layers
  - Backpropagation for weight updates
  - Training with cost monitoring

```python
# Example usage
nn = NeuralNetwork(784, 16)  # 784 inputs, 16 hidden nodes
predictions, cost = nn.train(X_train, Y_train, iterations=1000)
```

#### 3. Deep Neural Network Class (`16-deep_neural_network.py` to `28-deep_neural_network.py`)
- **Multi-Layer Deep Networks**: Support for arbitrary number of hidden layers
- **Advanced Features**:
  - Multiple hidden layers with configurable sizes
  - Multiclass classification support
  - Different activation functions (sigmoid, tanh)
  - Save/load functionality for trained models
  - Enhanced training with visualization

```python
# Example usage
# Binary classification
deep_nn = DeepNeuralNetwork(784, [128, 64, 1])

# Multiclass classification (10 classes)
deep_nn = DeepNeuralNetwork(784, [128, 64, 10])

# With tanh activation in hidden layers
deep_nn = DeepNeuralNetwork(784, [128, 64, 10], activation='tanh')
```

### Key Features by Version

| File | Description | Key Features |
|------|-------------|--------------|
| `0-neuron.py` | Basic neuron initialization | Weight initialization, private attributes |
| `1-neuron.py` | Forward propagation | Sigmoid activation function |
| `2-neuron.py` | Cost calculation | Logistic regression cost |
| `3-neuron.py` | Model evaluation | Binary predictions, accuracy |
| `4-neuron.py` | Gradient descent | Single optimization step |
| `5-neuron.py` | Basic training | Simple training loop |
| `6-neuron.py` | Enhanced training | Verbose output, cost monitoring |
| `7-neuron.py` | Visualization | Training cost graphs |
| `8-15-neural_network.py` | Neural networks | Hidden layer, backpropagation |
| `16-23-deep_neural_network.py` | Deep networks | Multiple layers, advanced training |
| `24-one_hot_encode.py` | Label encoding | Convert labels to one-hot format |
| `25-one_hot_decode.py` | Label decoding | Convert one-hot back to labels |
| `26-deep_neural_network.py` | Model persistence | Save/load trained models |
| `27-deep_neural_network.py` | Multiclass support | Softmax activation, categorical loss |
| `28-deep_neural_network.py` | Activation functions | Configurable sigmoid/tanh activations |

### Utility Functions

#### One-Hot Encoding (`24-one_hot_encode.py`)
Converts numeric class labels to one-hot encoded format for multiclass classification.

```python
Y_encoded = one_hot_encode([0, 1, 2, 1], 3)
# Result: [[1, 0, 0, 0],
#          [0, 1, 0, 1], 
#          [0, 0, 1, 0]]
```

#### One-Hot Decoding (`25-one_hot_decode.py`)
Converts one-hot encoded predictions back to numeric class labels.

```python
labels = one_hot_decode(one_hot_matrix)
# Result: [0, 1, 2, 1]
```

## 🎯 Mathematical Foundations

### Advanced Linear Algebra
- **Determinants**: Matrix determinant calculation
- **Minors & Cofactors**: Matrix minor and cofactor computation
- **Adjugate Matrices**: Adjugate matrix calculation
- **Matrix Inverse**: Inverse matrix computation
- **Definiteness**: Positive definite matrix testing

### Bayesian Probability
- **Likelihood**: Probability likelihood calculations
- **Intersection**: Joint probability computations
- **Marginal Probability**: Marginal distribution calculations
- **Posterior**: Bayesian posterior probability

### Calculus
- **Derivatives**: Partial and total derivatives
- **Integrals**: Definite and indefinite integration
- **Optimization**: Gradient-based optimization methods

### Convolutions and Pooling
- **Convolution Operations**: 2D convolutions for image processing
- **Pooling**: Max and average pooling operations
- **Padding**: Various padding strategies

## 🚀 Getting Started

### Prerequisites

```bash
# Required packages
pip install numpy matplotlib
```

### Basic Usage

1. **Train a Simple Neuron**:
```python
from supervised_learning.classification import Neuron
import numpy as np

# Load your data
X = np.random.randn(784, 1000)  # 1000 samples, 784 features
Y = np.random.randint(0, 2, (1, 1000))  # Binary labels

# Create and train neuron
neuron = Neuron(784)
predictions, cost = neuron.train(X, Y, iterations=1000)
print(f"Final cost: {cost}")
```

2. **Train a Deep Neural Network**:
```python
from supervised_learning.classification import DeepNeuralNetwork

# Create deep network
deep_nn = DeepNeuralNetwork(784, [128, 64, 32, 10])  # 4 layers

# Train the network
predictions, cost = deep_nn.train(X, Y, iterations=5000, 
                                 alpha=0.1, verbose=True, graph=True)

# Save the trained model
deep_nn.save("my_trained_model")

# Load the model later
loaded_model = DeepNeuralNetwork.load("my_trained_model.pkl")
```

3. **Multiclass Classification**:
```python
# For multiclass (e.g., 10 classes)
Y_multiclass = one_hot_encode(labels, 10)  # Convert to one-hot
deep_nn = DeepNeuralNetwork(784, [128, 64, 10])  # 10 output neurons
predictions, cost = deep_nn.train(X, Y_multiclass)
predicted_labels = one_hot_decode(predictions)  # Convert back to labels
```

## 📊 Features

### Activation Functions
- **Sigmoid**: `σ(x) = 1 / (1 + e^(-x))` - Default for hidden layers
- **Tanh**: `tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))` - Alternative for hidden layers  
- **Softmax**: Used for output layer in multiclass classification

### Cost Functions
- **Binary Cross-Entropy**: For binary classification
- **Categorical Cross-Entropy**: For multiclass classification

### Optimization
- **Gradient Descent**: Basic optimization algorithm
- **Learning Rate Control**: Configurable learning rates
- **He Initialization**: Proper weight initialization for deep networks

### Training Features
- **Verbose Output**: Real-time training progress
- **Cost Visualization**: Training cost graphs
- **Model Persistence**: Save/load trained models
- **Evaluation Metrics**: Accuracy and cost reporting

## 🔧 Code Quality

All code follows PEP 8 style guidelines and has been formatted using:
```bash
pycodestyle --statistics --count .
autopep8 --in-place --aggressive --aggressive *.py
```

## 📈 Performance

The implementations are optimized for educational purposes and include:
- Vectorized operations using NumPy
- Efficient matrix multiplications
- Numerical stability considerations
- Memory-efficient implementations

## 🤝 Contributing

This repository is part of the ALU Machine Learning curriculum. All implementations are built from scratch without using high-level ML frameworks to ensure deep understanding of the underlying mathematics.

## 📄 License

This project is part of academic coursework at ALU (African Leadership University).

## 📚 References

- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- Pattern Recognition and Machine Learning by Christopher Bishop
- The Elements of Statistical Learning by Hastie, Tibshirani, and Friedman

---

**Note**: This repository contains educational implementations. For production use, consider frameworks like TensorFlow, PyTorch, or scikit-learn.
