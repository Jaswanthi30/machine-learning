Data Preprocessing
1.Dropping Unnecessary Columns:
Removed "company_id", "company_name", "last_funding_date", and "hiring_roles" due to: 
•	company_id" is just an identifier. 
•	company_name" is redundant and ineffective for prediction. 
•	"last_funding_date" is a date column that was not converted to numerical characteristics. 
•	The model was simplified by ignoring the categorical "hiring_roles" field. 

2.Handling Categorical Features

•	LabelEncoder() converts text categories to numerical labels for Industry.
•	Consistent encoding across training and test sets reduces errors resulting from unidentified information. 


3.Handling Missing Values
Numerical Columns: The values that are missing have been substituted with the median value within each column. 
• Missing values in numerical attributes have been replaced with the median value to prevent data loss as well as outliers.
• To maintain consistency, the test sample uses the training data's median values.

Feature Engineering:
•	We explicitly define features by excluding is_hot_lead (the target variable) as well as remove unnecessary columns. Although rudimentary, other feature design strategies that might boost model performance include: 
•	 Establishing a standard numerical characteristics can increase stability of models
•	 Feature Interactions: Integrating preexisting features, including as funding per employee. 

Model Selection
  Two models:
RandomForestClassifier
•	Effectively manages numerical as well as categorical information. 
•	Effectively manages values that are missing as well as skewed data.
•	Added class_weight="balanced" to account for any category imbalances. 

XGBoostClassifier (Hyperparameter Tuned)

•	RandomizedSearchCV optimizes the number of estimators, learning rate, maximum depth, and subsample. 
•	Evaluated utilizing F1-score to balance precision and recall. 
•	Considering a dataset consisting of 20,000 training samples and 5,000 test samples, the technique that is most effective is to employ advanced models for machine learning such as XGBoost or Random Forest instead of deep learning.
•	XGBoost is especially well-suited for structured tabular data, as it improves performance by handling missing values, managing feature priority, and optimizing utilizing gradient boosting. 
•	It is computationally efficient, performs well with medium-sized datasets, and provides strong generalization via hyperparameter tweaking. While deep learning works well with unstructured data like photographs or text, it requires far larger datasets and more computer resources to avoid overfitting. 
•	Given these constraints, a well-tuned XGBoost model with hyperparameter optimization using RandomizedSearchCV is the best solution, offering an increased F1-score, robust projections and interpretability—all crucial for business decision-making in identifying high-potential leads.

Model Training & Evaluation
1️.Train-Test Split
•	80% training, 20% validation (stratified to maintain class distribution).
2️ .Model Training
Hyperparameters:
•	n_estimators=100: Uses 100 decision trees for robust predictions.
•	random_state=42: Ensures reproducibility.
•	class_weight="balanced": Adjusts weights to handle potential class imbalance.

3.Evaluation Metrics
Model Evaluation:
• Multiple categorization metrics are used to evaluate the model, exposing various aspects of performance.
F1-Score (Primary Metric)
• Balances recall and precision, which makes it appropriate for imbalanced datasets with few hot leads.
Formulation: 
    F1=2×Precision+RecallPrecision×Recall
• A higher F1-score indicates more accurate identification of hot leads.     

Accuracy
• Measures average accuracy, but might be misleading when data is skewed. 
•	For instance, forecasting all 0s with an accuracy rate of 90% seems ineffective if 90% of the companies are not hot leads.

Precision
• Analyzes the accuracy of predicted hot leads. 
• Excellent precision reduces false positives (incorrectly categorizing a company as a hot lead).

Recall
     • Counts whether numerous hot leads were correctly discovered. 
      • Excellent recall results in fewer false negatives (missing hot leads).


Confusion matrix:
The confusion matrix visualizes how well the model performed by displaying reliable and unreliable forecasts for "Hot Lead" and "Not Hot Lead." The heatmap clearly shows True Positives, True Negatives, False Positives, and False Negatives, making it simple to evaluate categorization efficiency. An successful model will have higher values along the diagonal (correct predictions), whereas misclassifications (off-diagonal values) highlight chances of development.

  Prediction & Submission
•	 Model makes predictions on the test dataset.
•	 Predictions are saved in submission.csv in the required format:
company_id,is_hot_lead
COMP_000001,1
COMP_000002,0
•	 File is saved as "submission.csv" for easy submission.


