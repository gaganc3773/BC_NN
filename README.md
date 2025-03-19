# **Binary Image Classification (Cat vs. Non-Cat) Using Fully Connected Neural Network**
## **Overview**
This project implements a **Fully Connected Neural Network (FCNN)** for binary image classification to distinguish between **cat and non-cat images**. The model processes images, extracts features, and classifies them using **deep learning** techniques.

## **Dataset**
- The Dataset consists of images categorized as `Cat` and `Non-Cat`.
- Images are `resized`, `normalized`, and `flattened` before feeding into the neural network.

## **Model Architecture**
### **$1.$ Input Processing**
- Loading the `Data` from `Data_sheet`
- Conversion of the `Data` into `Tensors`.
- Reshaping `4d tensor Data` into a `2d tensor Data`. ( Basically `Flattening` the image data)
- Normalizing the `pixel values` of the `flattened image`.( making the values to be in the range of `0 to 1`)
### **$2.$ Neural Network Structure**
- `Input Layer`: Flattened image vector.  
- `Hidden Layers`: Fully connected layers with `ReLU activation`.  
- `Output Layer`: Single neuron with `Sigmoid activation` for binary classification.  

The network is mathematically represented as:  
    $Y_{\theta}(x) = \sigma(W_n \cdots \text{ReLU}(W_2 \cdot \text{ReLU}(W_1 x + b_1) + b_2) + b_n)$

where:  
- **$x$** → Flattened image input  
- **$\theta(W, b)$** → Weights and biases
- **$\sigma$** → Sigmoid activation for binary classification  

### **$3.$ Loss Computation**
The model is optimized using `Binary Cross-Entropy Loss`:

$L = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log (1 - \hat{y}^{(i)}) \right]$

where:  
- **$y^{(i)}$** → True label (1 for cat, 0 for non-cat)  
- **$\hat{y}^{(i)}$** → Model's predicted probability  


### **$4.$ Backward Propagation & Optimization**
- Use `SGD(Stochastic gradient descent)` to update weights $\theta$.
- The optimization step follows:  
  $\theta = \theta - \eta \frac{\partial \mathcal{L}}{\partial \theta}$  
  where **$\eta$** is the learning rate.

### **$5.$ Training Process**
- Train the model on the labeled dataset.
- Monitor training loss and accuracy.
- performance using precision, recall, and accuracy


## **Features**
- **Preprocessing**: `Image resizing`, `normalization`, and `flattening.`
- **Deep Learning Model**: `FCNN` with `multiple hidden layers.`
- **Optimization**: Uses `binary cross-entropy loss` with `SGD` optimizer.
- **Evaluation**: Computes accuracy, precision, and recall metrics.

## **References**
- **Andrew NG**, *"Neural Networks and Deep Learning"*(`coursera`)



