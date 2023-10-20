# Customer Churn Prediction using Artificial Neural Networks 

### Objective:
To build a neural network model to accurately predict whether banking customers will churn (leave) based on their profile data like demographics, account details, etc. This helps identify at-risk customers for retention programs.

### Key Steps:

1. Data preprocessing - loading CSV, label encoding categorical variables, one-hot encoding, scaling  
2. Creating a sequential ANN model with dense layers and dropout for regularization
3. Compiling and fitting the model on the preprocessed train and test splits
4. Making predictions on new customer data and evaluating model performance 

### The main challenges were:

1. Dealing with mixed data types like numeric, categorical, and text features 
2. Addressing class imbalance as the majority of customers do not churn
3. Optimizing hyperparameters like layers, units, and activations for best accuracy
4. Interpreting model predictions to derive useful customer retention insights

### This provided valuable hands-on experience:
- Applying deep learning to a real-world business problem  
- Devising evaluation metrics to objectively measure model quality
- Gaining proficiency with Keras API and leveraging TensorFlow

The core modeling skills can now be transferred to other classification domains. Overall, this project enhanced my abilities in end-to-end machine learning model building and application of neural networks.

### How each of the aforementioned challenges was tackled or dealt with:

1. **Mixed data types:**
- Label encoding was used to convert categorical 'Gender' to numeric (X[:,2]=le.fit_transform(X[:,2]))
- One-hot encoding handled multi-class 'Geography' (ct.fit_transform(X)). This represented countries as binary vectors.
- No text features in this dataset. Normalization handled other numeric types.

2. **Class imbalance:** 
- Models generally perform poorly on imbalanced classes. Over-sampling could be used to duplicate minority class examples.
- Using accuracy alone as a metric would mask poor minority class prediction. Confusion matrix helps identify true/false positives and negatives.

3. **Model optimization:**
- Started with a simple 2 hidden layer network, gradually adjusted the number of units, activations, dropouts, etc.  
- Tried additional convolutional/LSTM layers since sequence/images were unavailable.
- Used callback functions like EarlyStopping to prevent overfitting.
- Permutation feature importance helped identify impactful predictors.

4. **Prediction interpretation:**  
- Studied relationships between features and targets via visualization.  
- Identified customer profiles most/least likely to churn based on predictions.
- Used model to simulate retention programs - if changes are made profile is unlikely to churn.

This practical project involved experimenting with different techniques and analyzing results through various evaluation metrics to develop a robust and well-performing model. Significant domain knowledge was also required to apply learnings effectively.

