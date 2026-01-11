## 📂 Project Structure & Scripts

This project consists of a modular pipeline for training a facial emotion recognition model and evaluating demographic fairness and bias. Each script is responsible for a clearly defined step in the workflow.

### Data Preparation

- **`make_fairface_test_only.py` → `data/fairface_test_only.csv`**  
  Generates a CSV containing only the test split, aligned with FairFace-style demographic evaluation.

- **`export_fer_images.py` → `data/fer_images/`**  
  Converts FER2013 CSV entries into image files for model training and evaluation.

- **`dataset_audit.py`**  
  Provides exploratory analysis and summary statistics of the dataset, including class distributions and potential imbalances.

---

### Model Training

- **`vgg_fer.py`**  
  Defines the CNN architecture (VGG-style) used for facial emotion recognition.

- **`train_fer.py` → `test_predictions.csv`**  
  Trains the CNN on the FER2013 dataset **without demographic attributes** and generates predictions on the test set.

---

### Demographic Inference

- **`demographics_deepface.py` → `outputs/deepface_demographics.csv`**  
  Uses DeepFace to infer demographic attributes (e.g., gender, race proxies) for FER2013 images.

- **`merge_demographics.py` → `outputs/merged_predictions_with_demographics.csv`**  
  Merges model predictions with inferred demographic attributes to enable group-based evaluation.

---

### Fairness & Bias Evaluation

- **`evaluate_bias_template.py`**  
  Template for bias evaluation logic and metrics.

- **`evaluate_fairness.py`**  
  Evaluates model performance across demographic groups, computing per-group accuracy and recall to identify disparities.

- **`analyze_predictions.py`**  
  Generates confusion matrices and summary statistics from `test_predictions.csv`.

---

## 🔁 End-to-End Workflow

1. Prepare datasets and images  
2. Train CNN model on FER2013  
3. Generate test predictions  
4. Infer demographic attributes using DeepFace  
5. Merge predictions with demographics  
6. Evaluate fairness and bias across groups  

---

## 🚫 Notes

- Model weights and large intermediate files are intentionally excluded from version control.
- Demographic labels are inferred proxies and used solely for fairness analysis, not ground truth.

## 📜 License & Usage Modification: Not permitted.

Redistribution: Only allowed with proper attribution and without any changes to the original files.

Commercial Use: Only with prior written consent.

📌 Attribution All credits for the creation, design, and development of this project go to:

Andre Menezes 📧 Contact: andremenezes231@hotmail.com 🌐 Website: https://andremenezes.dev

If this project is used, cited, or referenced in any form (including partial code, design elements, or documentation), you must provide clear and visible attribution to the original author(s).

⚠️ Disclaimer This project is provided without any warranty of any kind, either expressed or implied. Use at your own risk.

📂 File Integrity Do not alter, rename, or remove any files, directories, or documentation included in this project. Checksum or signature verification may be used to ensure file authenticity.

© 2025 Andre Menezes. All Rights Reserved.
