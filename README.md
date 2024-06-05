Dataset link: https://www.kaggle.com/datasets/sooyoungher/smoking-drinking-dataset

Proposed Approach: A feed-forward neural network architecture comprising five layers is instantiated. The initial layer serves as the input layer, incorporating 23 neurons to represent features from
the dataset. Subsequently, the second, third, and fourth layers function as hidden layers, encompassing 16, 8, and 4 neurons, respectively. Rectified Linear Unit (ReLU) is employed as the
activation function in these hidden layers. The fifth layer, acting as the output layer, comprises a single neuron tasked with predicting binary outputs (0 or 1). The Sigmoid function
serves as the activation function for the output layer. To introduce regularization to the dense layer weights, a kernel_regularizer parameter utilizing L2 regularization penalty
is applied, with a regularization strength set to 0.001. The algorithm employs a batch size of 200 and utilizes the Adam optimizer with a learning rate of 0.0002. For training purposes, 
the binary cross-entropy loss function is employed, fitting the nature of binary classification tasks. The architectural representation of the model is visually depicted in Figure. 
Additionally, the training loss versus validation loss plot for the proposed neural network architecture is illustrated in Figure., providing insights into the convergence and performance of the model during training.
This neural network design is intentionally structured to accommodate complex relationships within the data. The choice of ReLU activation in hidden layers promotes nonlinear learning, 
while the Sigmoid activation in the output layer facilitates binary classification tasks. The incorporation of L2 regularization aids in preventing overfitting by penalizing large weights, 
contributing to a more generalized model. The choice of the Adam optimizer, coupled with an appropriately tuned learning rate, enhances the efficiency of the optimization process during training. 
The use of batch-wise updates, indicated by the batch size of 200, further optimizes the convergence process. This architecture, embodied in Fig.2., is a manifestation of thoughtful design choices 
tailored to the characteristics of the dataset and the objectives of the binary classification task. The training loss versus validation loss plot in Fig.3. serves as a diagnostic tool, 
providing insights into the model’s performance and generalization capabilities during the training process.

### Architecture of the model
![architecture](https://github.com/SudhanshuGulhane/Smoking-History-Predictor/assets/50482460/34d2e7d2-20a5-4979-b2d7-036e51c20aca)

### Results
![cross-validation-nn](https://github.com/SudhanshuGulhane/Smoking-History-Predictor/assets/50482460/a563cc63-630f-4cfb-801a-60d03a61808f)

### Plots
![loss_plot](https://github.com/SudhanshuGulhane/Smoking-History-Predictor/assets/50482460/21daaeba-57a0-4dea-93c4-aa957e82d5f6)

### Hyperparameter tuning
![image](https://github.com/SudhanshuGulhane/Smoking-History-Predictor/assets/50482460/bd9be91e-6c9b-432c-9326-f453d347b5de)

