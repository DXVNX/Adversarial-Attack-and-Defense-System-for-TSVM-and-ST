# Adversarial-Attack-and-Defense-System-for-TSVM-and-ST

# Adversarial Defense Systems for SVM-Based Classifiers

This repository contains implementations of adversarial defense mechanisms for Support Vector Machine (SVM) based classifiers, specifically focusing on semi-supervised learning scenarios. The project compares a baseline SVM with two advanced models:

- **AD-TSVM**: Adversarial Defense Twin Support Vector Machine (a previously established method).
- **AD-ST**: A novel Adversarial Defense System adapted from AD-TSVM, incorporating enhancements for improved robustness against finite attacks.

The models are evaluated on the [Spambase dataset](https://archive.ics.uci.edu/dataset/94/spambase) from the UCI Machine Learning Repository. The focus is on defending against adversarial attacks in classification tasks, such as spam detection.

## Key Features
- Implementation of a finite attack model to simulate adversarial perturbations.
- Semi-supervised learning using labeled and unlabeled data.
- Hyperparameter tuning via RandomizedSearchCV.
- Performance evaluation metrics: Accuracy, Precision, Recall, F1-Score.
- Visualizations: Bar plots for metrics comparison and t-SNE for data/prediction visualization under clean and attacked conditions.
- Novel contribution: AD-ST is an original adaptation of the adversarial defense from AD-TSVM, extending it to a new framework (AD-ST) for enhanced robustness.

## Requirements
- Python 3.8+
- Libraries:
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - ucimlrepo
  - scipy

## Usage
The project consists of two Jupyter notebooks:

1. **AD_TSVM(BaselineSVM).ipynb**:
   - Implements the AD-TSVM model as a baseline.
   - Loads the Spambase dataset via UCI.
   - Trains a standard SVM and the AD-TSVM model.
   - Applies a finite attack model to simulate adversarial examples.
   - Evaluates performance on clean and attacked data.
   - Generates metric plots and t-SNE visualizations.

2. **AD_ST.ipynb**:
   - Implements the novel AD-ST model (adapted from AD-TSVM).
   - Similar workflow: data loading, model training, attack simulation, evaluation, and visualizations.
   - Highlights the robustness improvements in AD-ST under adversarial conditions.


## Models Explanation

### Finite Attack Model
- Simulates adversarial attacks by perturbing samples based on the model's decision boundary.
- Parameters: `f_attack` (attack factor), `epsilon` (perturbation bound).
- Only attacks correctly classified samples to flip decisions minimally.

### AD-TSVM (Baseline)
- Extends Twin Support Vector Machine (TSVM) with adversarial defense.
- Uses semi-supervised learning: Trains on labeled data + pseudo-labeled unlabeled data.
- Incorporates adversarial loss, pseudo-label loss, and finite loss in optimization.
- Hyperparameters tuned for C1/C2 (regularization), lambda_adv (adversarial weight), etc.

### AD-ST (Novel Adaptation)
- A new model adapting the adversarial defense mechanism from AD-TSVM.
- Modifications include custom loss functions (e.g., TSVM loss, adversarial loss, pseudo loss, finite loss).
- Designed for improved handling of finite attacks in semi-supervised settings.
- This is an original contribution, building on prior AD-TSVM work to create a more robust variant.

Both models use linear kernels by default but can be extended.

## Dataset
- **Spambase**: 4601 instances, 57 features (word frequencies, etc.), binary classification (spam/non-spam).
- Fetched via `ucimlrepo.fetch_ucirepo(id=94)`.
- Preprocessing: Standardization, split into labeled/unlabeled/test sets (unlabeled ratio ~70%).

## Acknowledgments
- Based on UCI Spambase dataset.
- Inspired by prior work on AD-TSVM; AD-ST is a novel extension.
- Uses scikit-learn for core ML components.
