from data import get_image_data
from network import MLP
import matplotlib.pyplot as plt
import numpy as np
import random

def main():
    (train_images, train_labels), (test_images, test_labels) = get_image_data()

    mlp = MLP()
    mlp.fit(10, 0.08, train_images, train_labels)

    index = random.randint(0, test_images.shape[0])

    predicted = mlp.predict(test_images[index])
    expected = np.argmax(test_labels[index])
    print(f"Predicted digit: {predicted} - Expected digit: {expected}")

    plt.imshow(test_images[index].reshape(28,28), cmap='gray')
    plt.axis('off')
    plt.show()
    
if __name__ == "__main__":
    main()