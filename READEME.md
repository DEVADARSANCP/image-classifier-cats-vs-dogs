# 🐶🐱 Cats vs Dogs Image Classification using Deep Learning

This project demonstrates a complete pipeline for **binary image classification** using deep learning. It classifies images into **cats** or **dogs** using Convolutional Neural Networks (CNNs) and also showcases **transfer learning** with MobileNetV2. A **Streamlit web app** is included for live image prediction.

---

## 📁 Project Structure
cats_vs_dogs.ipynb # Jupyter notebook for training the models
app.py # Streamlit web app to test predictions
cats_vs_dogs_model.h5 # First model (used in app)
image_classifier_model2.h5 # Second model (lower accuracy)

## ✅ Features

- Image preprocessing and augmentation
- CNN training from scratch
- Transfer learning using MobileNetV2
- Evaluation using accuracy, confusion matrix, and classification report
- Streamlit-based web interface to upload and classify images
- Saved `.h5` model for deployment

---

## 🧰 Libraries Used

- `tensorflow` / `keras` – Model building and training
- `numpy`, `matplotlib` – Data manipulation and visualization
- `Pillow` – Image processing
- `streamlit` – Front-end web app

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/DEVADARSANCP/cats-vs-dogs-classifier.git
   cd cats-vs-dogs-classifier

2. Install dependencies:
    pip install tensorflow streamlit numpy Pillow matplotlib
    
3. Run the app:
    streamlit run app.py

🧠 Models Used
Custom CNN (used in the final app)

MobileNetV2 (transfer learning for comparison)

Both models were trained and evaluated on a dataset of cat and dog images, with data augmentation techniques to improve generalization.

📈 Evaluation Metrics
Accuracy Score

Loss Curve

Confusion Matrix

Classification Report (Precision, Recall, F1-score)

💾 Output
The trained models are saved as .h5 files and loaded directly in the Streamlit app for predictions.

📌 Notes
The dataset directory (e.g., /content/data/train) must be structured with two folders: /cats and /dogs.

This project can be improved using hyperparameter tuning, better augmentation, or deploying the app online (e.g., on Streamlit Cloud or Hugging Face Spaces).

## ⚠️ Model File Note

The trained model file (`.h5`) is not included in this repository because it exceeds GitHub's 100MB upload limit.

To use the application:
- You can retrain the model using the provided notebook (`cats_vs_dogs.ipynb`)
- Or request the model file via email or download it from the following link:

🔗 [Download Trained Model via Google Drive](https://drive.google.com/file/d/1tpNFXEImBFTqRZMAUOHbfzaHEgpRtgAL/view?usp=drive_link)
