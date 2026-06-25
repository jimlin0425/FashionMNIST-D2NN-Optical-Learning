# Fashion MNIST Classification: ML vs CNN vs Optical Neural Network (D2NN)

This repository contains a comparative study exploring three distinct approaches to image classification on the Fashion MNIST dataset. We benchmarked traditional machine learning, spatial deep learning, and a physics-driven optical computing simulation.

## 🚀 Project Overview

The core objective of this project is to implement and evaluate a **Diffractive Deep Neural Network (D2NN)** using TensorFlow/Keras, and compare its performance against standard baseline models. By encoding images into optical phase modulations and simulating wave diffraction, we explore the potential of zero-power, light-speed optical computing.

### Models Implemented:
1. **Random Forest (1D Baseline):** A traditional machine learning ensemble method.
2. **Convolutional Neural Network (CNN):** A standard 2D spatial feature extractor.
3. **Diffractive Deep Neural Network (D2NN):** A custom-built architecture simulating optical wave propagation, utilizing Euler's formula for phase mask modulation.

## 🔬 Technical Highlights

* **Custom Phase Mask Layer:** Implemented a physics-informed Keras layer using `tf.complex` and Euler's formula ($e^{i\theta} = \cos\theta + i\sin\theta$) to simulate optical phase modulation.
* **Learned Diffraction Operator:** Conducted an ablation study comparing strict Fourier Transform (`fft2d`) with a Dense layer approximation for free-space wave propagation. The data-driven Dense layer ("Learned Diffraction") successfully overcame GPU memory/sync bottlenecks for small matrices (28x28) and achieved higher accuracy by optimizing the spatial transformation matrix.

## 📊 Results and Evaluation

*Please see the generated evaluation charts below:*

### Comprehensive Performance Comparison
<img width="1189" height="690" alt="image" src="https://github.com/user-attachments/assets/f974983a-4d22-484b-b9d9-a1881a0edf6c" />

### Confusion Matrices
* **Random Forest:** <img width="920" height="790" alt="image" src="https://github.com/user-attachments/assets/d049ed33-6c5f-49f5-a306-af65f0cf1508" />

* **CNN:** <img width="920" height="790" alt="image" src="https://github.com/user-attachments/assets/62117b94-6c27-417b-8c3f-c7fd03fd205d" />

* **D2NN:** <img width="920" height="790" alt="image" src="https://github.com/user-attachments/assets/ad9c5234-0314-44c4-8230-688ef5eda24a" />


**Key Findings:**
* **CNN achieved the highest overall accuracy (~90%):** The 2D spatial convolution kernels effectively captured hierarchical visual features in the fashion images.
* **Random Forest provided a strong baseline (87.6%):** As a traditional ML method, it performed well but could not capture spatial relationships as effectively as the CNN.
* **D2NN demonstrated the viability of optical computing (~79%):** While achieving a lower accuracy than the baselines, the optical neural network successfully proved the effectiveness of phase modulation and diffraction-based feature extraction, showcasing the unique properties of optical interference patterns for specific categories.

## 🛠️ How to Run

1. Clone this repository.
2. Install the required dependencies:
```bash
   pip install -r requirements.txt
