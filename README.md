# 🖼️ Multi-Model Image Segmentation App using Streamlit & YOLO

This repository contains a fully functional **Image Segmentation Web Application** built using **Streamlit**, **YOLO**, **OpenCV**, and **Python**.  
The app supports **multiple pretrained segmentation models**, allowing users to upload an image and visualize segmented output with smooth, transparent mask overlays.

---

## 🚀 Features

🔥 Supports multiple segmentation models:
- 🧠 Brain Tumor Segmentation  
- 🛣 Roads Segmentation  
- 🌿 Leaf Disease Segmentation  
- 🧍 Person Segmentation  
- 🕳 Pothole Segmentation  
- ⚡ Cracks Segmentation  

🔥 Smooth transparent mask blending  
🔥 Clean & modern UI (custom background image)  
🔥 Uses YOLO segmentation models (Ultralytics)  
🔥 Real-time overlay generation with custom class colors  
🔥 Easy to extend for additional models  

---


## 🧠 Supported Segmentation Models
| Model ID | Task | Weights File | Classes |
|---------|------|--------------|---------|
| 1 | Brain Tumor Segmentation | `brain_tumor.pt` | bg, Tumor |
| 2 | Road Segmentation | `road.pt` | bg, Road |
| 3 | Crack Detection | `cracks.pt` | bg, Cracks |
| 4 | Leaf Disease Segmentation | `leaf_disease.pt` | bg, Disease |
| 5 | Person Segmentation | `person.pt` | bg, Person |
| 6 | Pothole Detection | `pothole.pt` | bg, Pothole |

---

## 📥 Download Model Weights  
Place all weights inside the `Weights/` folder.
https://drive.google.com/drive/folders/19ObW9wy7dKRTJfxgX4gCLxO-6hXRKOfy?usp=sharing
