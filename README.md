# Classification of images into two classes using Convolutional Neural Network (CNN)
Image Classifcation using Python and Keras
This project trains a model that can classify images of cars and planes using Deep Learning.
## Model Architecture
### Input Layer

`Input Shape` : (224, 224, 3) for RGB images (height, width, channels).
### Convolutional Layers

`Conv2D Layer 1`: 32 filters, kernel size: (2, 2).

`Activation function`: ReLU.

`Purpose` : Extract local features from the input image.

`MaxPooling2D Layer 1`:

`Pool size`: (2, 2).

`Purpose`: Reduce spatial dimensions while retaining important features.

`Conv2D Layer 2`:
32 filters, kernel size: (2, 2).

`Activation function`: ReLU.

`Purpose`: Further refine feature extraction.

`MaxPooling2D Layer 2`:

`Pool size`: (2, 2).

`Conv2D Layer 3`:
64 filters, kernel size: (2, 2).

`Activation function`: ReLU.

`Purpose`: Capture more complex patterns in the data.

`MaxPooling2D Layer 3`:

`Pool size`: (2, 2).

`Flatten Layer`

`Purpose`: Flatten the feature maps into a 1D vector for input into the fully connected layers.

`Fully Connected Layers`

`Dense Layer 1`:
64 neurons.
`Activation function`: ReLU.

`Purpose`: Learn complex patterns and relationships in the data.

`Dropout Layer`:

`Dropout rate`: 0.5.

`Purpose`: Reduce overfitting by randomly setting a fraction of input units to 0 during training.

`Dense Layer 2 (Output Layer)`:
1 neuron.

`Activation function`: Sigmoid.

`Purpose`: Output a probability score for binary classification.

`Compilation`
The model is compiled using the following settings:

`Loss Function`: Binary Cross-Entropy – measures the difference between predicted and true labels for binary classification tasks.

`Optimizer`: RMSProp – an adaptive learning rate optimization algorithm.

`Metrics`: Accuracy – measures the percentage of correct predictions.
