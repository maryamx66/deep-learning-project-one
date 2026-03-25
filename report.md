# MNIST Digit Classification

## Introduction

This report details the development of a deep learning model for handwritten digit classification using the **MNIST dataset**.
The primary objective of this project was to implement foundational neural network methodologies while focusing on the rationale behind our architectural and training decisions. Throughout the pipeline, specific choices regarding data preprocessing, the "depth over width" network topology, and simultaneous regularization techniques were applied to optimize performance and prevent overfitting.

## 1. Data Acquisition and Preprocessing

The data preparation was the first step of this project to ensure stable inputs and rigorous evaluation boundaries.

- Dataset Setup: We utilized the MNIST dataset. To bypass potential server blocking and SSL verification errors during the download, a custom HTTPS context and user-agent were configured.

- Data Transformation: Input images were preprocessed using transforms.ToTensor(). This applies min-max scaling, converting raw pixel values (0-255) into normalized tensors (0-1). This normalization is essential for stabilizing weight updates and allowing the optimizer to converge more efficiently.

- Dataset Splitting: The data was divided into 60% training, 20% validation, and 20% testing sets using a random split. The three-way split ensures the model's hyperparameters can be tuned on the validation set without leaking information from the final, unseen testing data, preventing data leakage.

- Data Loading Strategy: PyTorch's DataLoader was implemented with a batch size of 64. The training data was shuffled per epoch, while validation and testing sets remained unshuffled. Shuffling the training data breaks sequential bias, forcing the model to generalize rather than memorize the order of the inputs.

## 2. Model Architecture

The model was structured as a Multilayer Perceptron (MLP) tailored for classifying 28x28 grayscale images into 10 distinct digit classes.

- Input Layer: The 2D 28x28 image tensors are flattened into a 1D vector of 784 neurons.

- Hidden Layers: The architecture utilizes two hidden layers, stepping down from 128 neurons to 64 neurons. This explicitly follows a "depth > width" principle. Prioritizing a deeper network allows the model to learn complex, hierarchical features step-by-step, which is more effective than using a single, excessively wide layer.

- Activation Functions: The Rectified Linear Unit (ReLU) activation function was applied after both hidden layers. The ReLU fucntion introduces necessary non-linearity to the network while mitigating the vanishing gradient problem.

- Output Layer: The final layer maps the 64 features to 10 output classes. A Softmax activation was intentionally omitted here because the raw logits are passed directly to CrossEntropyLoss, which handles the Softmax calculations more efficiently internally.

## 3. Training Methodology and Hyperparameter Tuning

The training pipeline was built to maximize learning efficiency while aggressively combating overfitting.

- Loss Function and Optimizer: We utilized CrossEntropyLoss and the Adam Optimizer. Cross-entropy is the standard methodology for multi-class classification, and Adam provides adaptive learning rates that generally yield faster convergence than standard stochastic gradient descent.

- Hyperparameters: The Adam optimizer was initialized with a learning rate of 0.001, and the maximum number of training epochs was capped at 30.

- Simultaneous Regularization Methods: To robustly prevent the model from overfitting the training data, two distinct regularization methods were executed simultaneously:
  1. Dropout: A Dropout layer with a probability of 0.5 (p=0.5) was applied during the forward pass after the ReLU activations of both hidden layers. This prevents overfitting by randomly zeroing out neuron activations, forcing the network to learn redundant and robust representations.

  2. Weight Decay: L2 Regularization was applied by passing a weight decay factor of 1e-5 directly into the Adam optimizer. This penalizes excessively large weights, keeping the model simpler and less prone to memorizing noise.

- Model Selection and Early Stopping: The training loop monitored the validation loss at the end of every epoch. An early stopping mechanism was implemented with a patience of 5 epochs to halt training when the model stops evolving, saving computational resources and preventing late-stage overfitting. Whenever the validation loss reached a new low, a deep copy of the model's weights (best_model_zehra.pth) was saved to ensure we retained the best-performing iteration.
