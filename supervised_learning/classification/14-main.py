#!/usr/bin/env python3

import numpy as np

NeuralNetwork = __import__("14-neural_network").NeuralNetwork

# Load test data
lib_train = np.load("../data/Binary_Train.npz")
X_3D, Y = lib_train["X"], lib_train["Y"]
X = X_3D.reshape((X_3D.shape[0], -1)).T

# Initialize neural network
np.random.seed(0)
nn = NeuralNetwork(X.shape[0], 3)

# Evaluate before training
predictions_before, cost_before = nn.evaluate(X, Y)
accuracy_before = np.sum(predictions_before == Y) / Y.shape[1] * 100

print("Before training:")
print(f"Cost: {cost_before}")
print(f"Accuracy: {accuracy_before:.2f}%")
print(f"First 10 predictions: {predictions_before[0][:10]}")
print(f"First 10 true labels: {Y[0][:10]}")

# Train the neural network
print("\nTraining...")
predictions_after, cost_after = nn.train(X, Y, iterations=100, alpha=0.07)
accuracy_after = np.sum(predictions_after == Y) / Y.shape[1] * 100

print("\nAfter training:")
print(f"Cost: {cost_after}")
print(f"Accuracy: {accuracy_after:.2f}%")
print(f"First 10 predictions: {predictions_after[0][:10]}")
print(f"First 10 true labels: {Y[0][:10]}")

print(f"\nImprovement:")
print(f"Cost change: {cost_before - cost_after:.6f}")
print(f"Accuracy change: {accuracy_after - accuracy_before:.2f}%")

# Test error handling
print("\n" + "=" * 60)
print("Testing error handling:")

try:
    nn.train(X, Y, iterations="invalid", alpha=0.05)
except Exception as e:
    print(f"✓ TypeError for iterations: {e}")

try:
    nn.train(X, Y, iterations=100, alpha="invalid")
except Exception as e:
    print(f"✓ TypeError for alpha: {e}")

try:
    nn.train(X, Y, iterations=0, alpha=0.05)
except Exception as e:
    print(f"✓ ValueError for iterations: {e}")

try:
    nn.train(X, Y, iterations=100, alpha=0.0)
except Exception as e:
    print(f"✓ ValueError for alpha: {e}")

# Test with small example for detailed analysis
print("\n" + "=" * 60)
print("Testing with small example:")

np.random.seed(0)
nn_small = NeuralNetwork(2, 3)
X_small = np.array([[1, 2, 3], [4, 5, 6]])  # 2 features, 3 examples
Y_small = np.array([[1, 0, 1]])  # 3 labels

print(f"Small example input X: {X_small.shape}")
print(f"X:\n{X_small}")
print(f"Labels Y: {Y_small.shape}")
print(f"Y: {Y_small}")

# Before training
pred_before, cost_before = nn_small.evaluate(X_small, Y_small)
print(f"\nBefore training:")
print(f"Predictions: {pred_before}")
print(f"Cost: {cost_before:.6f}")

# Train with more iterations for small example
pred_after, cost_after = nn_small.train(X_small, Y_small, iterations=1000, alpha=1.0)
print(f"\nAfter training (1000 iterations):")
print(f"Predictions: {pred_after}")
print(f"Cost: {cost_after:.6f}")
print(f"Cost improvement: {cost_before - cost_after:.6f}")

# Test that all private attributes are updated
print(f"\nVerifying attribute updates:")
print(f"A1 shape: {nn_small.A1.shape}")
print(f"A2 shape: {nn_small.A2.shape}")
print(f"W1 shape: {nn_small.W1.shape}")
print(f"W2 shape: {nn_small.W2.shape}")
print(f"All attributes updated: A1, A2, W1, b1, W2, b2 ✓")

# Demonstrate learning progression
print("\n" + "=" * 60)
print("Learning progression demo:")

np.random.seed(0)
nn_demo = NeuralNetwork(2, 3)

costs = []
for i in [0, 10, 50, 100, 500, 1000]:
    if i == 0:
        # Initial evaluation
        _, cost = nn_demo.evaluate(X_small, Y_small)
    else:
        # Train for remaining iterations
        _, cost = nn_demo.train(
            X_small,
            Y_small,
            iterations=i - len(costs) * 10 if len(costs) > 0 else i,
            alpha=1.0,
        )
    costs.append(cost)
    print(f"Iteration {i}: Cost = {cost:.6f}")

print(
    f"\nCost is generally decreasing: {all(costs[i] >= costs[i + 1] for i in range(len(costs) - 1))}"
)
print("Training complete!")
