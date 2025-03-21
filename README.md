Healthcare Fraud Detection: Supervised Learning Approach
Introduction
Healthcare fraud costs the U.S. healthcare system an estimated $68 billion annually. Our objective was to detect fraudulent providers using machine learning by leveraging supervised learning approaches.

This project evaluates Logistic Regression, Gradient Boosting (GBC), XGBoost, LightGBM, and a Stacked Model (XGBoost + LightGBM) to determine the most effective fraud detection strategy.

EDA Challenges & Data Merging Issues
The train dataset lacked direct provider-level fraud indicators, making data merging complex.
We focused on inpatient and outpatient datasets, using claim-level records.
Engineered Features:
Claim Ratios
Visit Frequency
Total Reimbursed Amounts
Possible Data Issues:
Extreme values across multiple features suggest potential data fabrication.
Unrealistic billing patterns (e.g., excessive claims in short periods) hint at synthetic or altered records.
These irregularities reinforce the need for robust feature engineering and supervised models.
Fraud vs. Non-Fraud Distribution
Significant Class Imbalance:
Most providers are labeled as 'Non-Fraud' (0)
Fraudulent providers (1) make up a small percentage of the dataset
Key Takeaway: Class imbalance requires careful model tuning to optimize fraud detection.
Total Reimbursed by Fraud Status
Fraudulent providers receive significantly higher reimbursements than non-fraudulent providers.
Median reimbursement for fraudulent providers is much higher.
Clear separation between fraud and non-fraud in the log scale.
Extreme outliers in fraudulent cases suggest potential overbilling or suspicious claims.
Supervised Learning Approaches
We tested multiple supervised models:

Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	91.5%	N/A	<50%	N/A
Gradient Boosting (GBC)	92.3%	76%	56%	N/A
XGBoost (Tuned)	92.33%	57.63%	67.33%	62.10%
LightGBM (Tuned)	91.68%	54.26%	69.31%	60.87%
Stacked Model	93.35%	69.33%	51.49%	59.09%
Model Insights
Tuned Models
Fine-tuned hyperparameters:
max_depth, learning_rate, n_estimators, colsample_bytree, subsample
LightGBM had higher recall (better at identifying fraud).
XGBoost had a better balance between precision and recall.
Takeaway: LightGBM detects more fraud (higher recall), XGBoost provides a more balanced approach.
Threshold Adjustment (0.65)
Increased precision (fewer false positives) but lower recall (missed fraud cases).
XGBoost showed a better precision-recall balance.
LightGBM still flagged more fraudulent providers but with more false positives.
Stacked Model (XGBoost + LightGBM)
Stacking improved precision (69.33%) but recall dropped to 51.49%.
Best suited when false positives are more damaging than false negatives.
Stacked models perform better in balanced datasets, so additional fine-tuning may be needed.
Performance Evaluation: ROC & PR Curves
ROC Curve (AUC Scores)
Model	AUC Score
LightGBM	0.936
XGBoost	0.935
Stacked Model	0.913
Key Takeaway: XGBoost and LightGBM perform nearly identically, while the Stacked Model shows a slight drop.

Precision-Recall (PR) Curve
Model	Average Precision (AP) Score
XGBoost	0.712
LightGBM	0.705
Stacked Model	0.647
Key Takeaway:

XGBoost and LightGBM show nearly identical performance.
Stacked Model underperforms in precision-recall balance, confirming limited benefit from ensembling.
Feature Importance Analysis
Key Findings
Total Reimbursed is the strongest predictor for both models.
Claim amount variability matters:
Avg OP Claim and Std IP Claims Per Month suggest fraudulent providers exhibit inconsistent billing patterns.
LightGBM emphasizes visit frequency (e.g., Inpatient Visits, Total Visits).
XGBoost prioritizes reimbursement-based factors.
SHAP Analysis
SHAP for XGBoost
SHAP values represent log odds:
Positive SHAP values → Increase fraud likelihood.
Negative SHAP values → Push prediction toward non-fraud.
Total Reimbursed has the highest impact.
Total Visits and Avg OP Claim are major factors in predicting fraud.
SHAP for LightGBM
Total Reimbursed remains the strongest predictor.
Avg OP Claim & Total Visits significantly impact predictions.
Outpatient Visit Frequency & Claim Amount Variability play a role in fraud likelihood.
Conclusion & Next Steps
Key Takeaways
Supervised models were the most effective for fraud detection.
XGBoost & LightGBM provided strong recall and precision.
Stacked Model improved precision but slightly reduced recall.
Threshold tuning (0.65) optimized fraud detection by reducing false positives.
Top fraud indicators: Total Reimbursed, Total Visits, Claim Ratios.
SHAP Analysis confirmed feature importance and model interpretability.
Future Work
Enhance Recall: Improve fraud detection through advanced feature selection.
Explore Unsupervised Learning: Implement autoencoders or anomaly detection for fraud pattern discovery.
Deploy the Model: Integrate into real-time claims processing systems.
Monitor Evolving Fraud Tactics: Continuously update model training with new fraud patterns.
Project Files
final_provider_dataset_enhanced.csv → Processed dataset with fraud labels.
Healthcare_Fraud_Detection_Presentation.pptx → PowerPoint presentation of findings.
model_performance_results.png → Confusion matrices, ROC curves, PR curves.
shap_feature_importance.png → SHAP analysis results for feature importance.
feature_importance_lgb_xgb.png → Feature importance comparison.

