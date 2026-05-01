# 🚑 Dynamic Modality Switching for Robust Emergency Vehicle Detection

## 📌 Overview

This project detects emergency vehicles using a **multimodal approach**:

* 👁️ Vision (YOLOv8 object detection)
* 🔊 Audio (sirens detection)
* 🔗 Fusion of both modalities for better accuracy

It dynamically switches between audio and vision signals to improve detection in challenging conditions like noise or poor lighting.

---

## ⚙️ Features

* YOLOv8-based object detection
* Audio classification using spectrogram features
* Context-aware modality switching
* Multimodal fusion strategy
* Performance evaluation metrics

---

## 🗂️ Project Structure

```
├── data/                  # Dataset
├── Emergency_Vehicles/   # Images & labels
├── outputs/              # Results
├── runs/                 # Training outputs
├── src/                  # Source code
├── weights/              # Model weights
├── main.py               # Main pipeline
├── requirements.txt      # Dependencies
├── vision.yaml           # YOLO dataset config
```

---

## 🚀 Installation

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Train the model

```bash
python main.py --train
```

### 🔹 Quick test (faster)

```bash
python main.py --train --epochs-vision 5 --epochs-audio 2
```

### 🔹 Evaluate model

```bash
python main.py --evaluate
```

---

## 🧠 Technologies Used

* Python
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* Librosa
* NumPy / Pandas

---

## 📊 Output

* Trained models saved in `runs/`
* Performance metrics in `outputs/`
* Graphs via `generate_performance_graph.py`

---

## ⚠️ Notes

* Runs on CPU (slower on Mac M1)
* For faster training, use GPU (CUDA supported systems)

---

## 📈 Future Improvements

* Real-time detection system
* Mobile deployment
* Better audio dataset

---

## 👨‍💻 Author

Vikash Yadav

