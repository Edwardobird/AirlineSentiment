# BERT Fine-Tuned Model for Airline Tweet Classification

This repository contains a Jupyter Notebook (`Train.ipynb`) for fine-tuning a BERT model on a custom dataset of airline tweets. The model classifies tweets as positive, neutral, or negative and achieves an accuracy of over **95%** on the test dataset.

---

## Features
- **Preprocessing**:
  - Cleans the text data (removes URLs, mentions, hashtags, punctuation, and extra whitespaces).
  - Encodes sentiment labels (`negative` as `0`, `neutral` as `1`, `positive` as `2`).
  - Balances the dataset using text augmentation with `SynonymAug` from the `nlpaug` library.
- **Model**:
  - Fine-tunes a pre-trained BERT model (`bert-base-uncased`) using the `transformers` library.
  - Implements a weighted classification metric for accuracy, precision, recall, and F1 score.
- **Evaluation**:
  - Uses a GPU (if available) to efficiently evaluate the model on a test dataset.
  - Achieves over **95% accuracy** on the test set.

---

## Prerequisites

### Required Libraries
Ensure the following Python libraries are installed:
- `transformers`
- `datasets`
- `pandas`
- `scikit-learn`
- `torch`
- `seaborn`
- `matplotlib`
- `nlpaug`

You can install these using `pip` or the appropriate package manager for your environment.

### Hardware Requirements
- A GPU is recommended for faster training and evaluation.

### Dataset
The dataset must be a CSV file with the following columns:
- `airline_sentiment`: The sentiment label (`negative`, `neutral`, `positive`).
- `text`: The tweet content.

---

## Usage Instructions

1. **Set Up Environment**:
   - Install the required libraries.
   - Ensure you have a compatible Python environment (Python 3.7 or later recommended).

2. **Prepare the Dataset**:
   - Update the `file_dir` variable in the notebook to point to your dataset path.
   - Ensure the dataset has the required columns.

3. **Run the Notebook**:
   - Open `Train.ipynb` in Jupyter Notebook or Jupyter Lab.
   - Execute the cells in order, ensuring no errors occur.

4. **Adjust Paths**:
   - Replace Colab-specific paths (`/content/drive/...`) with your local or cloud paths as required.

5. **Model Evaluation**:
   - Evaluate the model using the provided evaluation function to ensure expected accuracy.

---

## Example Outputs

### Metrics:
- **Accuracy**: Over 95%
- **Precision, Recall, F1 Score**: Weighted metrics computed for all classes.

### Visualization:
- Bar charts showing the distribution of the original and balanced datasets.

---

## Notes

- **Directory Paths**: Update all hardcoded paths in the notebook to your local setup.
- **Custom Dataset**: You can modify the dataset and labels to fine-tune BERT for other classification tasks.
- **Augmentation**: The notebook includes augmentation for balancing datasets with fewer samples in certain classes.

---

## License

This project is licensed under the MIT License. Feel free to use and modify it for your purposes.
