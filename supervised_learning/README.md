# Supervised Learning - Classification

This module contains complete implementations of neural networks from scratch, progressing from single neurons to deep neural networks with advanced features.

## � File Structure

### Neuron Implementations (Single Neuron)
- `0-neuron.py` - Basic neuron with initialization and private attributes
- `1-neuron.py` - Forward propagation with sigmoid activation
- `2-neuron.py` - Cost calculation using logistic regression
- `3-neuron.py` - Model evaluation with binary predictions
- `4-neuron.py` - Gradient descent optimization step
- `5-neuron.py` - Basic training loop implementation
- `6-neuron.py` - Enhanced training with verbose output
- `7-neuron.py` - Training with cost visualization graphs

### Neural Network Implementations (Single Hidden Layer)
- `8-neural_network.py` - Basic neural network initialization
- `9-neural_network.py` - Forward propagation through hidden layer
- `10-neural_network.py` - Cost calculation for neural networks
- `11-neural_network.py` - Evaluation with hidden layer
- `12-neural_network.py` - Gradient descent with backpropagation
- `13-neural_network.py` - Basic neural network training
- `14-neural_network.py` - Enhanced training with monitoring
- `15-neural_network.py` - Complete neural network with visualization

### Deep Neural Network Implementations (Multiple Hidden Layers)
- `16-deep_neural_network.py` - Deep network initialization
- `17-deep_neural_network.py` - Deep network forward propagation
- `18-deep_neural_network.py` - Deep network cost calculation
- `19-deep_neural_network.py` - Deep network evaluation
- `20-deep_neural_network.py` - Deep network gradient descent
- `21-deep_neural_network.py` - Basic deep network training
- `22-deep_neural_network.py` - Enhanced deep network training
- `23-deep_neural_network.py` - Complete deep network with all features

### Utility Functions
- `24-one_hot_encode.py` - Convert numeric labels to one-hot encoding
- `25-one_hot_decode.py` - Convert one-hot encoding back to numeric labels

### Advanced Features
- `26-deep_neural_network.py` - Model persistence (save/load functionality)
- `27-deep_neural_network.py` - Multiclass classification support
- `28-deep_neural_network.py` - Configurable activation functions (sigmoid/tanh)

### Test Files
- `0-main.py`, `14-main.py`, etc. - Test files for each implementation

## 🚀 Quick Start Examples

### 1. Single Neuron (Binary Classification)

```python
#!/usr/bin/env python3
import numpy as np
from Neuron import Neuron

# Create dataset
X = np.random.randn(784, 1000)  # 1000 samples, 784 features (like MNIST)
Y = (np.random.randn(1, 1000) > 0).astype(int)  # Binary labels

# Create and train neuron
neuron = Neuron(784)
predictions, cost = neuron.train(X, Y, iterations=5000, alpha=0.05, 
                                verbose=True, graph=True, step=100)

print(f"Final cost: {cost:.6f}")
print(f"Accuracy: {np.mean(predictions == Y) * 100:.2f}%")
```

### 2. Neural Network (Single Hidden Layer)

```python
#!/usr/bin/env python3
import numpy as np
from NeuralNetwork import NeuralNetwork

# Load data
X = np.random.randn(784, 1000)
Y = (np.random.randn(1, 1000) > 0).astype(int)

# Create neural network with 16 hidden nodes
nn = NeuralNetwork(784, 16)
predictions, cost = nn.train(X, Y, iterations=1000, alpha=0.1,
                            verbose=True, graph=True, step=100)

print(f"Final cost: {cost:.6f}")
```

### 3. Deep Neural Network (Multiple Hidden Layers)

```python
#!/usr/bin/env python3
import numpy as np
from DeepNeuralNetwork import DeepNeuralNetwork

# Create deep network: 784 inputs -> 128 -> 64 -> 32 -> 1 output
deep_nn = DeepNeuralNetwork(784, [128, 64, 32, 1])

# Train the network
predictions, cost = deep_nn.train(X, Y, iterations=2000, alpha=0.075,
                                 verbose=True, graph=True, step=100)

# Save the trained model
deep_nn.save("my_deep_model")
print("Model saved successfully!")
```

### 4. Multiclass Classification

```python
#!/usr/bin/env python3
import numpy as np
from DeepNeuralNetwork import DeepNeuralNetwork
from one_hot_encode import one_hot_encode
from one_hot_decode import one_hot_decode

# Create multiclass dataset (10 classes)
X = np.random.randn(784, 1000)
y_labels = np.random.randint(0, 10, 1000)  # Class labels 0-9
Y = one_hot_encode(y_labels, 10)  # Convert to one-hot

# Create network for multiclass: 784 inputs -> 128 -> 64 -> 10 outputs
deep_nn = DeepNeuralNetwork(784, [128, 64, 10])

# Train for multiclass classification
predictions, cost = deep_nn.train(X, Y, iterations=1000)

# Convert predictions back to class labels
predicted_labels = one_hot_decode(predictions)
accuracy = np.mean(predicted_labels == y_labels)
print(f"Multiclass accuracy: {accuracy * 100:.2f}%")
```

### 5. Using Different Activation Functions

```python
#!/usr/bin/env python3
from DeepNeuralNetwork import DeepNeuralNetwork

# Create networks with different activations
nn_sigmoid = DeepNeuralNetwork(784, [128, 64, 10], activation='sig')
nn_tanh = DeepNeuralNetwork(784, [128, 64, 10], activation='tanh')

print(f"Sigmoid network activation: {nn_sigmoid.activation}")
print(f"Tanh network activation: {nn_tanh.activation}")

# Train both networks
pred_sig, cost_sig = nn_sigmoid.train(X, Y, iterations=500)
pred_tanh, cost_tanh = nn_tanh.train(X, Y, iterations=500)

print(f"Sigmoid final cost: {cost_sig:.6f}")
print(f"Tanh final cost: {cost_tanh:.6f}")
```

### 6. Model Persistence (Save/Load)

```python
#!/usr/bin/env python3
from DeepNeuralNetwork import DeepNeuralNetwork

# Train a model
deep_nn = DeepNeuralNetwork(784, [128, 64, 10])
predictions, cost = deep_nn.train(X, Y, iterations=1000)

# Save the trained model
deep_nn.save("trained_model")  # Automatically adds .pkl extension

# Load the model later
loaded_model = DeepNeuralNetwork.load("trained_model.pkl")

if loaded_model is not None:
    print("Model loaded successfully!")
    # Use the loaded model for predictions
    new_predictions, new_cost = loaded_model.evaluate(X_test, Y_test)
else:
    print("Failed to load model")
```

## 🔧 Implementation Details

### Mathematical Foundations

#### Forward Propagation
```
For each layer l:
Z[l] = W[l] × A[l-1] + b[l]
A[l] = activation_function(Z[l])
```

#### Activation Functions
- **Sigmoid**: `σ(z) = 1 / (1 + e^(-z))`
- **Tanh**: `tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))`
- **Softmax**: `softmax(z_i) = e^(z_i) / Σ(e^(z_j))`

#### Cost Functions
- **Binary Cross-Entropy**: `J = -1/m Σ[y log(a) + (1-y) log(1-a)]`
- **Categorical Cross-Entropy**: `J = -1/m Σ Σ y_ij log(a_ij)`

#### Backpropagation
```
Output layer: dZ[L] = A[L] - Y
Hidden layers: dZ[l] = W[l+1]^T × dZ[l+1] × g'(Z[l])

Weight updates:
dW[l] = 1/m × dZ[l] × A[l-1]^T
db[l] = 1/m × sum(dZ[l])
W[l] = W[l] - α × dW[l]
b[l] = b[l] - α × db[l]
```

### Features by Version

| Version | Key Features |
|---------|--------------|
| **Neuron (0-7)** | Single neuron, sigmoid activation, binary classification |
| **NeuralNetwork (8-15)** | Single hidden layer, backpropagation |
| **DeepNeuralNetwork (16-23)** | Multiple hidden layers, advanced training |
| **Utilities (24-25)** | One-hot encoding/decoding for multiclass |
| **Advanced (26-28)** | Save/load, multiclass, configurable activations |

### Training Parameters

- **iterations**: Number of training iterations (default: 5000)
- **alpha**: Learning rate (default: 0.05)
- **verbose**: Print training progress (default: True)
- **graph**: Show cost visualization (default: True)
- **step**: Progress reporting interval (default: 100)

## 🎯 Use Cases

### Binary Classification
- Email spam detection
- Medical diagnosis (positive/negative)
- Image classification (cat/dog)

### Multiclass Classification
- Handwritten digit recognition (MNIST)
- Image classification (multiple categories)
- Text classification (multiple topics)

### Deep Learning Applications
- Computer vision tasks
- Natural language processing
- Pattern recognition

## ⚡ Performance Tips

1. **Data Preprocessing**: Normalize input data to [0,1] or [-1,1] range
2. **Learning Rate**: Start with 0.1, adjust based on convergence
3. **Network Architecture**: Start small, gradually increase complexity
4. **Activation Functions**: Try both sigmoid and tanh for hidden layers
5. **Training Monitoring**: Use verbose=True to watch convergence

## 🐛 Common Issues and Solutions

### Poor Convergence
- Reduce learning rate
- Increase number of iterations
- Check data normalization
- Try different activation functions

### Overfitting
- Reduce network size
- Add regularization (future enhancement)
- Use more training data

### Slow Training
- Increase learning rate (carefully)
- Reduce network complexity
- Use vectorized operations

## 📚 Learning Resources

These implementations are designed for educational purposes to understand:
- Forward propagation mechanics
- Backpropagation algorithm
- Gradient descent optimization
- Neural network architectures
- Deep learning fundamentals

## 🔄 Workflow

1. **Start Simple**: Begin with single neuron
2. **Add Complexity**: Progress to neural networks
3. **Go Deep**: Implement deep neural networks
4. **Add Features**: Include multiclass, save/load, etc.
5. **Experiment**: Try different architectures and parameters

---

**Note**: All implementations use only NumPy and basic Python libraries to ensure deep understanding of the underlying mathematics.
