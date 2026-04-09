# MNIST Digit Classification

## Introduction

This report details the development of a deep learning model for handwritten digit classification using the **MNIST dataset**.
The primary objective of this project was to implement foundational neural network methodologies while focusing on the rationale behind our architectural and training decisions. Throughout the pipeline, specific choices regarding data preprocessing, the "depth over width" network topology, and simultaneous regularization techniques were applied to optimize performance and prevent overfitting.

## 1. Data Acquisition and Preprocessing

The data preparation was the first step of this project to ensure stable inputs and rigorous evaluation boundaries.

- Dataset Setup: We utilized the MNIST dataset. To bypass potential server blocking and SSL verification errors during the download, a custom HTTPS context and user-agent were configured.

- Data Transformation: Input images were preprocessed using `transforms.ToTensor()`. This applies min-max scaling, converting raw pixel values (0-255) into normalized tensors (0-1). This normalization is essential for stabilizing weight updates and allowing the optimizer to converge more efficiently.

- Dataset Splitting: The data was divided into 60% training, 20% validation, and 20% testing sets using a random split. The three-way split ensures the model's hyperparameters can be tuned on the validation set without leaking information from the final, unseen testing data, preventing data leakage.

- Data Loading Strategy: PyTorch's `DataLoader` was implemented with a batch size of 64. The training data was shuffled per epoch, while validation and testing sets remained unshuffled. Shuffling the training data breaks sequential bias, forcing the model to generalize rather than memorize the order of the inputs.

## 2. Model Architecture

The model was structured as a Multilayer Perceptron (MLP) tailored for classifying 28x28 grayscale images into 10 distinct digit classes.

- Input Layer: The 2D 28x28 image tensors are flattened into a 1D vector of 784 neurons.

- Hidden Layers: The architecture utilizes two hidden layers, stepping down from 128 neurons to 64 neurons. This explicitly follows a "depth > width" principle. Prioritizing a deeper network allows the model to learn complex, hierarchical features step-by-step, which is more effective than using a single, excessively wide layer.

- Activation Functions: The Rectified Linear Unit (ReLU) activation function was applied after both hidden layers. The ReLU fucntion introduces necessary non-linearity to the network while mitigating the vanishing gradient problem.

- Output Layer: The final layer maps the 64 features to 10 output classes. A Softmax activation was intentionally omitted here because the raw logits are passed directly to `CrossEntropyLoss`, which handles the Softmax calculations more efficiently internally.

## 3. Training Methodology and Hyperparameter Tuning

The training pipeline was built to maximize learning efficiency while aggressively combating overfitting.

- Loss Function and Optimizer: We utilized `CrossEntropyLoss` and the Adam Optimizer. Cross-entropy is the standard methodology for multi-class classification, and Adam provides adaptive learning rates that generally yield faster convergence than standard stochastic gradient descent.

- Hyperparameters: The Adam optimizer was initialized with a learning rate of 0.001, and the maximum number of training epochs was capped at 30.

- Simultaneous Regularization Methods: To robustly prevent the model from overfitting the training data, two distinct regularization methods were executed simultaneously:
  1. Dropout: A Dropout layer with a probability of 0.5 (`p= 0.5`) was applied during the forward pass after the ReLU activations of both hidden layers. This prevents overfitting by randomly zeroing out neuron activations, forcing the network to learn redundant and robust representations.

  2. Weight Decay: L2 Regularization was applied by passing a weight decay factor of 1e-5 directly into the Adam optimizer. This penalizes excessively large weights, keeping the model simpler and less prone to memorizing noise.

- Model Selection and Early Stopping: The training loop monitored the validation loss at the end of every epoch. An early stopping mechanism was implemented with a patience of 5 epochs to halt training when the model stops evolving, saving computational resources and preventing late-stage overfitting. Whenever the validation loss reached a new low, a deep copy of the model's weights (`best_model_zehra.pth`) was saved to ensure we retained the best-performing iteration.

## 3.1 Optimization Techniques and Comparetive Analysis

To evaluate the impact of advanced training methodologies, we added four more optimization techniques and compared the baseline model against an the optimized variant incorporating; Batch Normalization, He Initialization, Learning Rate Scheduling, and Gradient Clipping.

### Baseline Model Analysis

The baseline model demonstrated successful learning, with both training and validation errors decreasing steadily. However, the training process was halted prematurely around Epoch 21 due to the early stopping mechanism. As the model approached a local minimum, the fixed learning rate (0.001) likely caused the optimizer to oscillate, preventing further improvements in validation loss and triggering the patience counter.

![Figure 1: Model Before Optimization](images/model-training-zehra1.jpeg)
_Figure 1: Model Before Optimization_

### Optimized Model Analysis

The optimized model exhibited a notably different and more stable training trajectory.

- Enhanced Stability: The validation loss curve in the optimized model is exceptionally smooth, particularly after Epoch 10. This increased stability is primarily attributed to Batch Normalization, which reduces internal covariate shift by normalizing layer inputs, and Gradient Clipping, which prevents volatile parameter updates.

- Deeper Convergence: Unlike the baseline, the optimized model successfully trained for the full 30 epochs without triggering early stopping. This was achieved through the implementation of Learning Rate Scheduling (Step Decay). By dropping the learning rate by a factor of 0.1 every 10 epochs, the optimizer was able to take smaller, more precise steps later in the training process, settling deeper into the minimum without oscillating out of it.

- Effective Initialization: The use of He (Kaiming) Initialization ensured that the initial weight scales were properly tailored for the ReLU activations, preventing vanishing signals in the early epochs and allowing the network to maintain a strong learning signal from the very first batch.

  ![Figure 2: Model After Optimization](images/train_optimized2.png)
  _Figure 2: Model After Optimization_

### Conclusion of Comparison

While both models successfully classified the digits without overfitting, the integration of these optimization techniques resulted in a more robust training pipeline. The optimized model benefited from smoother convergence and the ability to train longer, demonstrating the practical necessity of learning rate scheduling and normalization in deep learning architectures.

## 4. Evaluation and Performance Metrics

The final phase of the project involved a rigorous evaluation of the trained model using the isolated test dataset to determine its real-world generalization capabilities as well as a comparative analysis of the training optimizations.

### Model State Loading:

The evaluation script specifically loads the `best_model_zehra.pth` weights that were saved during the early-stopping phase of training. This ensures the final evaluation is performed on the optimal model state that achieved the lowest validation loss, rather than the potentially overfitted weights of the final training epoch.

### Deterministic Data Loading:

The test dataset DataLoader was configured with `shuffle=False`. Shuffling is unnecessary during the testing phase. Keeping the data sequential guarantees that the model's output predictions perfectly align with the true labels when computing evaluation metrics.

### Granular Classification Metrics:

Beyond calculating overall accuracy, the evaluation utilizes `sklearn.metrics.classification_report` to generate precision, recall, and F1-scores for each of the 10 digit classes. Since Overall accuracy can sometimes mask specific weaknesses, granular metrics reveal exactly which digits the model struggles to identify or frequently mislabels.

### Confusion Matrix Visualization:

A heatmap of the confusion matrix is generated and saved as `confusion_matrix.png` using the Seaborn library. This provides a clear, visual diagnosis of class overlaps. It allows us to quickly identify specific misclassification patterns. For example, whether the network commonly confuses the digit '4' with the digit '9'.

### Misclassification Analysis:

The script identifies the specific indices where predictions do not match the true labels and plots a grid of up to 16 misclassified images (`misclassified_samples.png`). Manually reviewing the errors helps determine if the model is failing on clear numbers, or if the mistakes are isolated to highly ambiguous, poorly written digits that even a human might struggle to classify.

## 4.1 Comparative Analysis of Optimization Techniques

To evaluate the impact of advanced training methodologies, we compared our baseline model against an optimized variant incorporating Batch Normalization, He Initialization, Learning Rate Scheduling (Step Decay), and Gradient Clipping.

### Baseline Model Convergence:

The baseline model demonstrated successful learning, but the training process was halted prematurely around Epoch 21 by the early stopping mechanism. As the model approached a local minimum, the fixed learning rate (0.001) caused the optimizer to oscillate, preventing further improvements in validation loss.

### Optimized Model Stability:

The optimized model exhibited a notably more stable training trajectory. The validation loss curve smoothed out significantly, primarily due to Batch Normalization reducing internal covariate shift and Gradient Clipping preventing volatile parameter updates.

### Deeper Convergence:

The optimized model successfully trained for the full 30 epochs without triggering early stopping. The Step Decay scheduler progressively shrank the learning rate, allowing the optimizer to settle deeper into the loss minimum without oscillating.

## 4.2 Final Results of Evaluation Metrics:

The final evaluation on the isolated test set yielded the following results:

### Baseline Model Accuracy: 97.14%

![](images/confusion-mat-before.png)

### Optimized Model Accuracy: 97.16%

![](images/confusion-mat-after.png)

While the absolute increase in accuracy is marginal (+0.02%), this is a known phenomenon when classifying the MNIST dataset using Multilayer Perceptrons. The baseline model was already capable of reaching the "accuracy ceiling" for this specific architectural complexity.

However, the added optimization techniques helped in the stability and reliability of the training process. The optimized model achieved this peak performance with a significantly smoother convergence trajectory and without prematurely stalling, proving that the network learned efficiently and robustly.

## Conclusion

This project successfully demonstrated the end-to-end development of a neural network for image classification. By systematically addressing data preprocessing, model architecture, and training regularization, we established a robust pipeline tailored to the MNIST dataset.

Our architectural decision to prioritize depth over width in the Multilayer Perceptron allowed the model to effectively learn complex, hierarchical representations of the handwritten digits. Furthermore, aggressively combating overfitting through the simultaneous application of Dropout layers, L2 regularization (weight decay), and an automated early stopping mechanism proved crucial for ensuring the model generalized well to unseen data.

Building upon this strong baseline, the integration of advanced training methodologies—specifically Batch Normalization, He Initialization, Learning Rate Scheduling, and Gradient Clipping—proved highly effective. While the baseline model had already approached the accuracy ceiling for this specific dataset, our comparative analysis demonstrated that these optimizations significantly enhanced the stability of the convergence trajectory and allowed the model to settle deeper into the loss minimum without oscillating.

Finally, our evaluation methodology provided actionable insights that extended beyond a simple accuracy metric. By generating a class-wise performance report, confusion matrices, and visualizations of misclassified samples, we established a framework to accurately diagnose the model's failure modes and identify overlapping class boundaries. These granular findings confirm that the methodologies selected from our coursework were both theoretically sound and practically effective for this specific classification problem.
