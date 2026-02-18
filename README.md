# 🫀 Valvular Heart Disease Prognosis — Flask + Hybrid ML Web Application

This is a comprehensive web-based application developed using Flask and advanced Machine Learning techniques that predicts the likelihood of valvular heart disease based on echocardiographic and clinical parameters. The application uses a hybrid model combining Random Forest and Support Vector Machine algorithms, trained on real-world data from Medocore Hospital, Srikakulam.

## 🎯 Project Overview

- **Dataset**: Real-world clinical data from Medocore Hospital, Srikakulam (1280 patient records)
- **Model**: Hybrid Machine Learning (Random Forest + SVM)
- **Accuracy**: 97.66% (cross-validated)
- **Features**: 13 echocardiographic and clinical parameters
- **Technology**: Flask, scikit-learn, pandas, numpy

## 🧠 How It Works

The application analyzes 13 key medical parameters including:
- Age, Gender, Aortic Valve area
- Left Atrium diameter, Ejection Fraction
- End Diastolic/Systolic Dimensions
- Interventricular Septum thickness
- Posterior Wall thickness
- Aortic root diameter
- Interatrial Septum assessment
- Right Ventricular Systolic Pressure
- Regional Wall Motion Abnormality

## 🧠 Model Information

- **Model Type**: Hybrid Ensemble (Random Forest + Support Vector Machine)
- **Accuracy**: 97.66% on test data
- **Training Data**: 1280 samples from Medocore Hospital, Srikakulam
- **Cross-validation**: 5-fold CV with mean accuracy of 98.28%

---

## 📁 Project Structure

```bash
heart_disease_predictor/
│
├── app.py
│   - Main Flask application with HybridModel class and prediction logic
│
├── actual/
│   ├── DataSet.csv
│   ├── Project_Final.ipynb
│   └── Project_Final.py
│   - Original dataset and Jupyter notebook with model training code
│
├── model/
│   ├── hybrid_model.pkl
│   └── scaler.pkl
│   - Trained hybrid model and data scaler
│
├── templates/
│   ├── base.html      - Base layout template
│   ├── index.html     - Home page with project overview
│   ├── predict.html   - Input form with clinical metadata
│   ├── result.html    - Prediction results display
│   ├── about.html     - Project information and model details
│   └── data.html      - Dataset information and features
│
├── static/
│   └── css/
│       └── style.css  - Application styling
│
├── requirements.txt   - Python dependencies
├── LICENSE           - Project license
└── README.md         - This file
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone the Repository

```bash
git clone https://github.com/sarathchandrasimma/Valvular-Heart-Disease-Prognosis.git
cd heart_disease_predictor
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

### 5. Access the Website

Open your browser and navigate to: `http://127.0.0.1:5000`

---

## 📊 Dataset Information

### Source
- **Hospital**: Medocore Hospital, Srikakulam
- **Type**: Real-world clinical echocardiographic data
- **Size**: 1280 patient records
- **Features**: 13 clinical parameters + target variable

### Clinical Parameters
| Feature | Clinical Significance | Reference Range |
|---------|----------------------|-----------------|
| AGE | Patient chronological age | Variable |
| GENDER | Biological sex classification | Binary (0=Female, 1=Male) |
| AORTIC VALVE | Cross-sectional valve area | 2.5-4.0 cm² |
| LEFT ATRIUM | Chamber diameter | <40 mm |
| EDD | Left ventricular diameter at diastole | 3.5-5.6 cm |
| ESD | Left ventricular diameter at systole | 2.0-4.0 cm |
| EF | Ventricular contractile function | 50-70% |
| IVS (D) | Septal wall thickness at diastole | 0.6-1.2 mm |
| PW (D) | Posterior wall thickness at diastole | 0.6-1.2 mm |
| AORTA | Proximal aortic root diameter | <30 mm |
| I.A.S | Structural integrity assessment | Binary (0=Normal, 1=Abnormal) |
| RVSP | RV pressure estimate | <35 mmHg |
| RWMA | Qualitative segmental kinesis assessment | Binary (0=Normal, 1=Abnormal) |

---

## 🔬 Model Performance

- **Training Accuracy**: 97.66%
- **Cross-validation Scores**: [0.98, 0.98, 0.98, 0.98, 0.99]
- **Mean CV Accuracy**: 98.28%
- **Standard Deviation**: 0.004

### Classification Report
```
              precision    recall  f1-score   support
0              0.95        0.95      0.95       56
1              0.98        0.98      0.98      200
accuracy       0.98        0.98      0.98      256
```

---

## 🚀 Features

- **Clinical Input Form**: User-friendly form with medical metadata and reference ranges
- **Real-time Prediction**: Instant heart disease risk assessment
- **Medical Guidance**: Personalized health recommendations based on predictions
- **Responsive Design**: Mobile-friendly web interface
- **Data Validation**: Proper input validation for clinical parameters

---

## 📧 Contact & Support

**Developer**: Sarath Chandra Simma
**Institution**: B.Tech Final Year Project
**Dataset Source**: Medocore Hospital, Srikakulam

For technical questions or collaboration:
- GitHub: [sarathchandrasimma](https://github.com/sarathchandrasimma)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⭐ Acknowledgments

- **Dataset**: Medocore Hospital, Srikakulam for providing real clinical data
- **Libraries**: scikit-learn, Flask, pandas, numpy
- **Inspiration**: Advancing healthcare through machine learning

---

# ⭐ Don't forget to **star** this repository if you found it helpful!
