Abstract

This project explores early breast cancer detection using feature selection with meta-heuristic algorithms and machine learning classifiers. Two public datasets were employed: WDBC (569 biopsy samples, 30 features) and Coimbra (116 clinical samples, 9 features).
I applied Wrapper Feature Selection guided by the Bat Algorithm (BA) and the Imperialist Competitive Algorithm (ICA), where the fitness function combines cross-validated accuracy with a sparsity bonus to encourage compact models.
Multiple classifiers were evaluated, including KNN, Logistic Regression, Random Forest, SVM, MLP, LDA, Naive Bayes, Decision Tree, AdaBoost, Softmax Regression, and GRU-SVM. 
Data preprocessing involved Min–Max normalization and stratified train/test splitting (60/40) to preserve class balance.
Evaluation metrics included Accuracy, Precision, Recall/Sensitivity, Specificity, F1-score, Kappa, MAE, and RMSE. 
Results show that both BA and ICA improve performance while reducing the number of features. BA achieved the highest accuracy (>94%) on WDBC, while ICA also produced strong and more balanced models.
The findings highlight the effectiveness of meta-heuristic feature selection in medical ML applications, improving not only predictive accuracy but also interpretability and clinical utility.
