# Classification-of-Documents-Using-Graph-Based-Features-and-KNN
This repository contains scripts for collecting, preprocessing, and modeling document data for classification into three categories: Fashion, Disease, and Sports. The project aims to demonstrate the process of document classification using machine learning techniques.

## Overview
Document classification is a fundamental task in natural language processing (NLP) and machine learning. It involves categorizing text documents into predefined classes or categories based on their content. In this project, we focus on classifying documents into three categories: Fashion, Disease, and Sports.

The project pipeline consists of the following steps:

- **Data Collection**: The data collection process involves scraping text data related to fashion, disease, and sports from various online sources. This is achieved using web scraping techniques to extract relevant content from websites, blogs, or online publications dedicated to each topic.
- **Data Preprocessing**: Once the data is collected, it undergoes preprocessing to clean and prepare it for analysis. This involves tasks such as tokenization, removing stopwords and punctuation, and converting text to lowercase. The preprocessed data is then ready for feature extraction.
- **Feature Extraction**: Feature extraction involves transforming the preprocessed text data into numerical representations suitable for machine learning algorithms. In this project, we use graph-based features such as distances between nodes and maximal common subgraph sizes to represent document structures.
- **Modeling**: With the extracted features, we build and train machine learning models for document classification. We employ algorithms like K-Nearest Neighbors (KNN) to classify documents into the predefined categories. The models are trained on a labeled dataset comprising examples from each category.
- **Evaluation**: The trained models are evaluated using a separate test dataset to assess their performance in classifying unseen documents. Evaluation metrics such as accuracy, precision, recall, and F1-score are computed, and a confusion matrix is generated to analyze the model's predictions.
## Project Structure
The project repository contains the following files:

1. **sportsCollection.py**: Script for scraping sports-related document data.
2. **fashionCollection.py**: Script for scraping fashion-related document data.
3. **diseaseCollection.py**: Script for scraping disease-related document data.
4. **Preprocessing.py**: Script for preprocessing the scraped document data.
5. **Modelling.py**: Script for building and training machine learning models for document classification.
## Usage
To use the project scripts, follow these steps:

- Clone the repository to your local machine.
- Run the data collection scripts (sportsCollection.py, fashionCollection.py, diseaseCollection.py) to collect document data for each category.
- Execute the preprocessing script (Preprocessing.py) to clean and prepare the collected data.
- Run the modeling script (Modelling.py) to build and train machine learning models for document classification.
- Evaluate the trained models using the provided evaluation metrics.ta, constructs classification models using algorithms like K-Nearest Neighbors (KNN), and evaluates the models' performance using metrics like accuracy and confusion matrices.
