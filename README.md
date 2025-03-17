# DA6401-Assignment01
# Neural Network Training and Evaluation

This document explains the training process of a neural network, focusing on optimization techniques, hyperparameters, and backpropagation mechanisms. The model is trained on the Fashion MNIST dataset and employs experiment tracking for performance evaluation.

---

## 1. Neural Network Training Process
The training process involves multiple steps, including forward propagation, loss computation, backpropagation, and weight updates. Each step is crucial in optimizing the model for better accuracy and generalization.

### Forward Propagation
- Input data is passed through multiple layers of neurons.
- Each neuron applies a weighted sum operation followed by an activation function.
- The output layer produces predictions based on learned weights.

### Loss Computation
- The difference between predictions and actual labels is measured using a loss function.
- Common loss functions include Cross-Entropy (for classification) and Mean Squared Error (for regression).

### Backpropagation
- Backpropagation updates weights based on the gradient of the loss function.
- It involves computing the derivative of the loss with respect to each weight.
- The gradients are propagated backward from the output layer to the input layer.

---

## 2. Optimizers
Optimizers play a crucial role in adjusting model parameters to minimize the loss function. Different optimizers have unique characteristics and are selected based on problem requirements.

### Types of Optimizers:
- **Gradient Descent:** Updates weights based on the average gradient across the entire dataset.
- **Stochastic Gradient Descent (SGD):** Updates weights for each sample, allowing faster convergence but adding variance.
- **Momentum-based SGD:** Introduces a momentum term to accelerate convergence and avoid local minima.
- **Adam (Adaptive Moment Estimation):** Combines momentum and adaptive learning rates, making it effective for most deep learning tasks.
- **RMSprop:** Uses a moving average of squared gradients to normalize updates, preventing extreme weight updates.

---

## 3. Hyperparameters
Hyperparameters control how the neural network learns and significantly affect its performance. Choosing appropriate values for these parameters requires careful tuning and experimentation.

### Key Hyperparameters:
- **Learning Rate:** Determines the step size in weight updates. A high value can lead to instability, while a low value may slow convergence.
- **Batch Size:** Defines the number of samples processed before weight updates. Larger batch sizes provide more stable gradients, whereas smaller sizes introduce noise but enhance generalization.
- **Number of Hidden Layers:** Affects the network’s ability to capture complex patterns. More layers improve expressiveness but increase the risk of overfitting.
- **Neurons per Layer:** Determines the computational power of each layer. Too many neurons may lead to redundancy, while too few may underfit.
- **Activation Functions:** Different functions (ReLU, Sigmoid, Tanh) impact the non-linearity and learning capacity of the network.
- **Weight Initialization:** Methods like Xavier or He Initialization prevent vanishing/exploding gradients, ensuring stable training.
- **Number of Epochs:** Defines the total passes over the dataset. Too few may result in underfitting, while too many can lead to overfitting.
- **Regularization Techniques:** Methods such as L2 regularization and dropout prevent overfitting by limiting weight growth or randomly deactivating neurons during training.

---

## 4. Hyperparameter Tuning
- **Bayesian Optimization:** Uses probabilistic models to find the best hyperparameters.
---


