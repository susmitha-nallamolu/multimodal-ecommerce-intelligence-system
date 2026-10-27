# Multimodal E-Commerce Intelligence System

A Deep Learning project that explores e-commerce product categorization using product text, images, and Transformer-based NLP.

## Project Overview

E-commerce platforms contain thousands of products that need to be organized into appropriate categories. This project compares multiple deep learning approaches for automated product categorization across **21 product categories**.

The project experiments with:

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Long Short-Term Memory (LSTM)
- CNN + LSTM multimodal model
- DistilBERT Transformer model

The models are evaluated using classification accuracy, macro F1-score, and other classification metrics.

## Project Objectives

- Explore and preprocess product titles, descriptions, and images.
- Build and evaluate different deep learning architectures.
- Compare traditional neural networks with Transformer-based text classification.
- Develop an interactive application using Streamlit.
- Display model evaluation results and product category predictions.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- DistilBERT
- CNN and LSTM
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Pillow

## Application Screenshots

### Product Category Prediction

![AI Product Category Prediction](screenshots/ai_prediction.png)

### Model Performance Comparison

![Five-Model Performance Comparison](screenshots/model_comparison.png)

### Product Catalogue

![Product Catalogue](screenshots/product_catalogue.png)

### Product Image Dataset

![Product Image Dataset](screenshots/training_images.png)

## Model Evaluation

The following results were obtained during evaluation on the project validation dataset.

| Model | Accuracy | Macro F1-score |
|---|---:|---:|
| ANN | 22.52% | 0.2027 |
| CNN | 8.45% | 0.0251 |
| LSTM | 9.40% | 0.0689 |
| CNN + LSTM | 22.51% | 0.1805 |
| DistilBERT | 60.45% | 0.5830 |

**Best-performing model:** DistilBERT achieved the highest validation accuracy among the evaluated models.

The results demonstrate the effectiveness of contextual Transformer-based text representations for this product categorization task.

## Application Features

- Text-based product category prediction using DistilBERT.
- Predicted category and confidence display.
- Model performance comparison dashboard.
- Product catalogue with search and category filtering.
- Classification reports and confusion matrices for model evaluation.

**Implementation note:** The live Streamlit prediction uses the text-only DistilBERT model. CNN, LSTM, and CNN + LSTM were evaluated separately; the current application does not perform live image-text multimodal fusion.

## Project Structure

```text
multimodal_ecommerce_intelligence/
├── app.py
├── README.md
├── requirements.txt
├── screenshots/
├── reports/
├── src/
│   ├── data_preprocessing/
│   └── training/
├── check_dataset.py
├── predict_product.py
└── visualize_dataset.py
```

## How to Run the Application

### 1. Clone the repository

```bash
git clone https://github.com/susmitha-nallamolu/multimodal-ecommerce-intelligence-system.git
cd multimodal-ecommerce-intelligence-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

**Note:** The application may also require the trained DistilBERT checkpoint and tokenizer files. These are not included in the repository by default. Follow the model setup instructions if you distribute the checkpoint separately.

## Evaluation Reports

The `reports/` directory contains selected evaluation artifacts, including:

- Classification reports
- Confusion matrices
- Training history
- Model comparison results and visualization
- Misclassification analysis

## Limitations

- CNN, LSTM, and CNN + LSTM achieved lower validation performance than DistilBERT in these experiments.
- The current live prediction pipeline uses text rather than image-text fusion.
- Model performance depends on the dataset, preprocessing, and training configuration.
- The trained model files and datasets are not included in the repository by default.

## Future Improvements

- Implement live multimodal image-text fusion.
- Improve image preprocessing and image classification performance.
- Explore further Transformer fine-tuning and model optimization.
- Deploy the application with a reproducible model-download workflow.

## Author

**Susmitha Nallamolu**

AI Engineer Aspirant | Deep Learning | NLP | Transformers | Python

GitHub: [susmitha-nallamolu](https://github.com/susmitha-nallamolu)
