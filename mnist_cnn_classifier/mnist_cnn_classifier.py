import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from keras.models import load_model

def main():
    # 1. Veri setini yükle
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # 2. Normalize et
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    # 3. Boyut ekle (28x28 → 28x28x1)
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)

    # 4. Etiketleri one-hot encode et
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # 5. Modeli kur
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    # 6. Derle
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    # 7. Eğit
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=5,
        batch_size=128
    )

    # 8. Başarımı çiz
    plt.plot(history.history['accuracy'], label='Eğitim Başarımı')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Başarımı')
    plt.title('Eğitim vs Doğrulama Başarımı')
    plt.xlabel('Epoch')
    plt.ylabel('Başarım')
    plt.legend()
    plt.tight_layout()
    plt.savefig("mnist_training_accuracy.png")
    plt.show()

    # 9. Modeli kaydet
    model.save("mnist_model.h5")
    print("Model başarıyla 'mnist_model.h5' olarak kaydedildi.")

if __name__ == "__main__":
    main()
