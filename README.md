# MNIST CNN görüntü sınıflandırma

---

## 🇬🇧 English README

```markdown
# MNIST Handwritten Digit Recognition – CNN Image Classification
```
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
```

# MNIST El Yazısı Rakam Tanıma – CNN ile Görüntü Sınıflandırma

Bu projede, el yazısı ile yazılmış rakamları içeren **MNIST** veri seti kullanılarak bir **Convolutional Neural Network (CNN)** modeli ile görüntü sınıflandırma yapılmıştır. Model 0'dan 9'a kadar olan rakamları tanımayı öğrenir. Proje, derin öğrenmeye başlangıç yapmak isteyenler için güçlü ve sade bir örnektir.

---

## 📌 Kullanılan Veri Seti

- **MNIST** veri seti, 28x28 boyutunda el yazısı rakam resimlerinden oluşur.
- 60.000 eğitim ve 10.000 test verisi içerir.
- `tensorflow.keras.datasets` üzerinden doğrudan yüklenir.

---

## 🔧 Kullanılan Teknolojiler ve Kütüphaneler

- Python 3.x
- TensorFlow / Keras
- NumPy
- Matplotlib

---

## 🚀 Proje Adımları

1. **Veri Yükleme ve Ön İşleme:**  
   - MNIST veri seti yüklendi.  
   - Görüntüler normalize edildi ve CNN için uygun şekle getirildi (28x28x1).  
   - Etiketler one-hot encoding formatına çevrildi.

2. **Model Mimarisi:**  
   - 2 konvolüsyonel katman + max pooling  
   - Flatten → Dense (128 nöron) → Dense (10 nöron, softmax)  
   - Aktivasyon fonksiyonları: `relu`, `softmax`  
   - Kayıp fonksiyonu: `categorical_crossentropy`  
   - Optimizasyon: `Adam`

3. **Model Eğitimi:**  
   - Epoch: 5  
   - Batch size: 128  
   - Eğitim ve doğrulama başarımı görselleştirildi.

4. **Model Kaydı:**  
   - Eğitilen model `.h5` dosyasına (`mnist_model.h5`) kaydedildi.

---

## 📁 Oluşturulan Dosyalar

| Dosya Adı                  | Açıklama                                     |
|---------------------------|----------------------------------------------|
| `mnist_cnn_classifier.py` | Ana Python dosyası                           |
| `mnist_model.h5`          | Eğitilmiş model                              |
| `mnist_training_accuracy.png` | Eğitim ve doğrulama başarımı grafiği     |

---

## 💻 Çalıştırma

```bash
python mnist_cnn_classifier.py
