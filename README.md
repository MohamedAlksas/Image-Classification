# 🐱🐶 Cat vs Dog Image Classifier

## 📌 Project Description

This project focuses on building an image classification model using Convolutional Neural Networks (CNNs) to distinguish between images of cats and dogs. The primary goal is to demonstrate how deep learning can be applied to binary classification tasks using a relatively simple CNN architecture

---

## ⚙️ Setup Instructions

### ✅ Prerequisites:
- Python 3.7+
- Basic understanding of neural networks and CNNs

### 🛠️ Installation Steps:

1. Clone the repository or download the project files.
2. Install dependencies:
   ```bash
   pip install tensorflow pillow matplotlib
   ```
3. Dataset directory structure:
   ```
   cd_dataset/
   ├── training_set/
   │   ├── cats/
   │   └── dogs/
   └── testing_set/
       ├── cats/
       └── dogs/
   ```

> ⚠️ Make sure images are correctly labeled and in `.jpg` or `.png` format. Corrupted images will be automatically removed.

---

## ▶️ How to Run

### Step-by-step Instructions:

1. **Remove Corrupted Images**  
   A built-in function checks and deletes unreadable or corrupted images from the dataset.

2. **Data Augmentation & Loading**  
   We use `ImageDataGenerator` to rescale, flip, shear, and zoom images to improve generalization.

3. **Model Training**
   - Model: Sequential CNN with 2 Conv2D + MaxPooling layers
   - Activation: ReLU and Sigmoid
   - Loss Function: Binary Crossentropy
   - Optimizer: Adam
   - Epochs: 25
   - Batch Size: 32

4. **Model Evaluation**
   - Accuracy and Loss are calculated on the test set using `.evaluate()`.

5. **Prediction Demo**
   - A random image from the test set is selected and predicted.
   - The prediction result and the image are displayed using `matplotlib`.

To run:
```bash
python cat_dog_classifier.py
```

---

## 📈 Training Results

| Metric           | Value (approx.)  |
|------------------|------------------|
| Training Accuracy| 90–95%           |
| Test Accuracy    | 85–90%           |
| Loss             | Stable around 0.2 after training |

Example Output:
```
Epoch 25/25
loss: 0.2143 - accuracy: 0.9120
Test Loss: 0.2321
Test Accuracy: 0.8875
```

The model performs well for a basic CNN architecture without transfer learning.

---

## 👥 Team Contributions

| ID         | Name                        | 
|------------|-----------------------------|
| 202203549  | Mohamed Adel Alksas         | 
| 202203193  | Mohamed Said Shalaby        |  
| 202203778  | Mohamed Magdy Ali           |  
| 202202378  | Mohamed Abd-Elaal Elsayes   |    
| 202203766  | Mohamed Tarek Hashad        | 


---