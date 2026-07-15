# Personalized Wake Word Engine

**Project Title:** Personalized Wake Word Engine  
**Name:** Yaswanth  
**Roll Number:** [Your Roll Number]  

## Problem Statement
Existing wake-word detection systems often struggle with accuracy and latency under noisy, real-world conditions. This project develops a deep learning-based wake word engine that accurately detects a predefined keyword ('marvin') from continuous audio in real time, setting a schedule or alarm upon detection.

## Architecture Diagram
![Architecture Diagram](docs/architecture.png)

## Dataset
- **Source:** [Google Speech Commands v0.02](https://research.google.com/audioset/download.html)
- **Description:** A dataset of spoken words designed to help train and evaluate keyword spotting systems. We are utilizing it to train the model to detect a specific wake word while ignoring background noise and other unknown words.

## How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the model:**
   ```bash
   python src/train.py
   ```
3. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

## Results
| Metric | Value |
|--------|-------|
| Accuracy | TBA |
| Precision | TBA |
| Recall | TBA |
| F1-Score | TBA |

*(Run the training script to populate metrics!)*

## Module Mapping
- **M1 (Neural Networks Basics):** Utilizes standard classification paradigms and loss functions (CrossEntropy).
- **M2 (Advanced Architectures):** Employs Convolutional Neural Networks (CNN) with 2D convolutions over MFCC features, Batch Normalization, and Adaptive Pooling.
- **M3 (Real-World Deployment):** Model is integrated into a Streamlit web application providing a real-time UI/UX for testing audio detection and scheduling alarms.

## References
1. Warden, P. (2018). "Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition." *arXiv preprint arXiv:1804.03209*.
2. Sainath, T. N., & Parada, C. (2015). "Convolutional neural networks for small-footprint keyword spotting." *Interspeech*.
3. Tang, R., & Lin, J. (2018). "Deep Residual Learning for Small-Footprint Keyword Spotting." *ICASSP*.
4. Hershey, S., et al. (2017). "CNN architectures for large-scale audio classification." *ICASSP*.
5. Zhang, Y., et al. (2017). "Hello Edge: Keyword Spotting on Microcontrollers." *arXiv preprint arXiv:1711.07128*.
