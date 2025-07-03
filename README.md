# MNIST CNN görüntü sınıflandırma

---

## 🇬🇧 English README

```markdown
# MNIST Handwritten Digit Recognition – CNN Image Classification

This project performs image classification on the **MNIST** handwritten digits dataset using a **Convolutional Neural Network (CNN)**. The model is trained to recognize digits from 0 to 9. It's a great starter project for those learning the basics of deep learning.

---

## 📌 Dataset Used

- The **MNIST** dataset consists of 28x28 grayscale images of handwritten digits.
- It includes 60,000 training and 10,000 test samples.
- Loaded directly from `tensorflow.keras.datasets`.

---

## 🔧 Technologies and Libraries

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🚀 Project Workflow

1. **Data Loading and Preprocessing:**  
   - MNIST data loaded  
   - Normalized pixel values  
   - Reshaped to (28, 28, 1)  
   - Labels one-hot encoded

2. **Model Architecture:**  
   - 2 Convolutional layers + max pooling  
   - Flatten → Dense (128 neurons) → Dense (10 neurons, softmax)  
   - Activation functions: `relu`, `softmax`  
   - Loss: `categorical_crossentropy`  
   - Optimizer: `Adam`

3. **Model Training:**  
   - Epochs: 5  
   - Batch size: 128  
   - Accuracy visualized via training plot

4. **Model Saving:**  
   - Trained model saved as `mnist_model.h5`

---

## 📁 Generated Files

| File Name                  | Description                                  |
|---------------------------|----------------------------------------------|
| `mnist_cnn_classifier.py` | Main Python file                             |
| `mnist_model.h5`          | Trained model                                |
| `mnist_training_accuracy.png` | Accuracy graph for training/validation   |

---

## 💻 Run the Project

```bash
python mnist_cnn_classifier.py

