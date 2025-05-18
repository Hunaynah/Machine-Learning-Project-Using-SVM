# Machine-Learning-Project-Using-SVM
Data Science Projects
Customer Churn Prediction Model using SVM 
Tools/Technologies Used: R, e1071, caret, ROCR, Data Preprocessing, Model Evaluation

Objective: Built a machine learning model to predict customer churn in a telecom company using a dataset with multiple customer attributes.

Data Cleaning & Feature Engineering: Performed data cleaning and feature selection to enhance model performance. Applied feature scaling to numeric columns for improved accuracy.

Handling Class Imbalance: Addressed the class imbalance issue using SMOTE (Synthetic Minority Over-sampling Technique), implemented through the caret package, to improve the model's prediction accuracy for the minority class (Churn).

Modeling: Developed a Support Vector Machine (SVM) classifier with a linear kernel for classification. 

Model Evaluation:

Evaluated the model using metrics like confusion matrix, accuracy, and AUC.

Generated an ROC curve and calculated AUC (Area Under the Curve) to assess the model’s ability to discriminate between churned and non-churned customers.

Achieved AUC = 0.76, demonstrating the model’s effective performance in predicting customer churn.

Outcome: The model successfully predicted customer churn and provided insights into critical factors affecting churn, helping to inform business strategies for customer retention.
