# 🎙️ Optimizing Neural Network-Based Speech Synthesis

**Author:** Colin Nguyen  
**Course:** Math 132A — Professor Paul J. Atzberger  
**Date:** March 18, 2025  

---

## Overview
This project explores how different optimization algorithms affect the **training efficiency and audio quality** of neural network–based speech synthesis systems.  

Using **Tacotron 2** and **WaveGlow** as baseline models, the goal was to test whether optimization techniques—**Stochastic Gradient Descent (SGD)**, **Adam**, and **Limited-memory BFGS (L-BFGS)**—can improve training speed, stability, and final speech quality.

---

## Motivation
Deep learning has revolutionized **text-to-speech (TTS)** systems, enabling models that generate audio nearly indistinguishable from human speech. However, these systems are **computationally expensive** and require long training times.

By applying and comparing different **optimization methods**, this project aims to identify which optimizer provides the best balance between **efficiency and audio fidelity**, making future speech generation models more practical and scalable.

**Key research questions:**
1. How do different optimizers affect convergence speed and stability?
2. How do they influence speech quality metrics such as **Mean Squared Error (MSE)** and **Mel Cepstral Distortion (MCD)**?
3. Which method achieves the best trade-off between efficiency and quality?

---

## Problem Formulation

Given a dataset \( D = \{(x_i, y_i)\}_{i=1}^N \), where:
- \( x_i \): input text embeddings  
- \( y_i \): target mel-spectrograms  

We minimize the **masked mean squared error (MSE)**:

\[
L_{MSE} = \frac{1}{N} \sum_i (y_i - \hat{y}_i)^2
\]

To evaluate synthesized speech, we compute **Mel Cepstral Distortion (MCD):**

\[
MCD = \frac{10}{\ln(10)} \sqrt{2 \sum_{i=1}^K (c_i - \hat{c}_i)^2}
\]

Where \( c_i \) and \( \hat{c}_i \) are ground-truth and predicted cepstral coefficients.  
(An MCD around **5** indicates human-like quality.)

---

## Theoretical Justification

While neural networks are **non-convex**, optimization theory still provides useful intuition:

- **SGD:** Converges to a local minimum under smoothness assumptions, but has high variance in gradient updates.  
- **Adam:** Adapts learning rates via first- and second-moment estimates, improving stability and convergence on non-convex surfaces.  
- **L-BFGS:** A quasi-Newton method that approximates the Hessian matrix to accelerate convergence when the loss surface is well-conditioned.

Assuming differentiability and locally convex regions, all three methods can theoretically reach meaningful minima—but their **empirical performance** varies significantly in high-dimensional TTS models.

---

## Methods

### Dataset
- **LJSpeech-1.1:** 13,100 short audio clips (≈24 hours) of a single speaker reading passages from books.  
- Mel-spectrograms extracted from audio samples.

### Model
- **Tacotron 2** as the baseline sequence-to-sequence architecture.  
- Loss: Masked MSE between predicted and true mel-spectrograms.  
- Validation metrics: MSE and MCD.

### Optimization Setup
- **Optimizers tested:** SGD, Adam, L-BFGS  
- **Epochs:** 100  
- **Batch size, learning rate, and dropout:** constant across runs  
- **Evaluation:**  
  - Convergence speed (loss reduction per epoch)  
  - Training stability (variance of loss)  
  - Final speech quality (MCD)  

---

## Results

| Optimizer | Validation Loss ↓ | MCD ↓ | Observations |
|------------|------------------:|------:|--------------|
| **Adam** | 4.60 | 2047.76 | Fastest and most stable convergence; lowest loss and distortion |
| **L-BFGS** | 9.06 | 2303.04 | Some acceleration, but unstable and erratic loss curve |
| **SGD** | 16.96 | 2303.04 | Slowest convergence, high variance, least effective overall |

**Key findings:**
- Adam achieved the best performance across all metrics.  
- L-BFGS showed instability due to noisy curvature approximations in high-dimensional space.  
- SGD suffered from slow progress and inconsistent updates.  

Despite limited compute (CPU-only training), the relative trends matched theoretical expectations.

---

## Interpretation

- **Adam:** Adaptive updates stabilize learning and handle complex gradient landscapes effectively.  
- **L-BFGS:** Potentially fast but unreliable without strong curvature information.  
- **SGD:** Prone to instability and poor convergence in deep, non-convex architectures.  

---

## Conclusion

This project demonstrates that **Adam** offers the best trade-off between **training efficiency** and **speech quality** for neural speech synthesis.  
Although computational constraints prevented full convergence, results highlight the importance of optimizer selection in large-scale TTS training.

**Limitations:**
- CPU-only hardware restricted training duration and fidelity.  
- Generated speech was unintelligible due to insufficient epochs (~100 vs. 1600–2000 typically needed).

**Future Work:**
- Run full training on GPU-equipped hardware.  
- Explore hybrid optimizers (e.g., Adam→L-BFGS schedule).  
- Add new metrics like **Signal-to-Noise Ratio (SNR)** and **Perceptual Evaluation of Speech Quality (PESQ)**.

---

## Technologies Used
- **Python**
- **PyTorch**
- **NumPy**, **Pandas**
- **Matplotlib**
- **Tacotron 2 / WaveGlow**
- **LJSpeech Dataset**
- **Adam**, **SGD**, **L-BFGS** optimizers

---
