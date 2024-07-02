# BBC News Classifier

## Description

The BBC News Classification project involves training and evaluating various Naive Bayes models on a dataset of BBC news articles. The project includes a Multinomial Naive Bayes model with additive smoothing and a Bernoulli Naive Bayes classifier. The dataset is composed of 2225 real news articles labeled as Business, Entertainment, Politics, Sport, and Tech. The project explores the impact of class imbalance on the model's performance and provides an analysis of the results.

## Features

- Multinomial Naive Bayes model with additive smoothing.
- Bernoulli Naive Bayes classifier.
- Analysis of class imbalance and its impact on model performance.
- Confusion matrices and accuracy metrics for model evaluation.

## Dataset

- **Source**: BBC News
- **Total Articles**: 2225
- **Labels**: Business (0), Entertainment (1), Politics (2), Sport (3), Tech (4)
- **Training Set**: 1668 articles
- **Test Set**: 557 articles
- **Preprocessing**: Each column indicates the number of occurrences of a given word for a document instance.

## Tech Stack

- **Programming Language**: Python
- **Libraries**: No additional libraries used, all implementations are self-contained.

## Installation

To run the project, follow these steps:

1. On the command line, navigate to the directory where the `main.py` file is located.
2. Type the following command to run the script:
   ```bash
   python main.py
   ```
3. No additional parameters are needed. The results will be printed on the command line after running the script.

## Results and Analysis

The project includes an analysis of the model's performance using confusion matrices and accuracy metrics. Key findings include:

- The Multinomial Naive Bayes model without smoothing fails to make accurate predictions, with an accuracy of 0.242:
  ```
  Confusion Matrix:
  [[135.   0.   0.   0.   0.]
   [102.   0.   0.   0.   0.]
   [ 98.   0.   0.   0.   0.]
   [134.   0.   0.   0.   0.]
   [ 88.   0.   0.   0.   0.]]
  ```

- The extended Multinomial Naive Bayes model with additive smoothing performs better by handling zero probabilities, achieving an accuracy of 0.977:
  ```
  Confusion Matrix:
  [[131.   0.   2.   0.   2.]
   [  0.  97.   0.   0.   5.]
   [  1.   0.  96.   0.   1.]
   [  0.   0.   1. 133.   0.]
   [  1.   0.   0.   0.  87.]]
  ```

- The Bernoulli Naive Bayes model, while effective, has slightly lower accuracy (0.932) compared to the smoothed Multinomial Naive Bayes model due to the binary nature of its features, leading to more misclassifications:
  ```
  Confusion Matrix:
  [[116.  10.   0.   0.   9.]
   [  0.  88.   2.   0.  12.]
   [  0.   0.  96.   0.   2.]
   [  0.   3.   0. 131.   0.]
   [  1.   0.   0.   0.  88.]]
  ```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contribution

Contributions are welcome! Please fork this repository and submit pull requests for any improvements or bug fixes.
