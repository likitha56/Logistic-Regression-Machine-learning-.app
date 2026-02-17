#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df = pd.read_csv('diabetes.csv')
display(df.head())


# In[3]:


print('DataFrame Info:')
df.info()

print('\nDescriptive Statistics:')
df.describe()


# In[4]:


import matplotlib.pyplot as plt
import seaborn as sns

# Get numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Plot histograms for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# In[5]:


import matplotlib.pyplot as plt
import seaborn as sns

# Get numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Plot box plots for each numerical column
for col in numerical_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
    plt.ylabel('Value')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calculate the correlation matrix
correlation_matrix = df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
plt.title('Correlation Matrix of Features')
plt.show()


# In[7]:


import numpy as np

# Define columns with potential implausible '0' values
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace '0' values with NaN in the specified columns
for col in columns_with_zeros:
    df[col] = df[col].replace(0, np.nan)

# Impute missing values (NaN) with the median of each column
for col in columns_with_zeros:
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

print("Implausible zero values replaced with NaN and then imputed with median for specified columns.")
print("Displaying the first 5 rows after imputation:")
display(df.head())


# In[8]:


import numpy as np

# Define columns with potential implausible '0' values
columns_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace '0' values with NaN in the specified columns
for col in columns_with_zeros:
    df[col] = df[col].replace(0, np.nan)

# Impute missing values (NaN) with the median of each column
for col in columns_with_zeros:
    median_val = df[col].median()
    df[col] = df[col].fillna(median_val)

print("Implausible zero values replaced with NaN and then imputed with median for specified columns.")
print("Displaying the first 5 rows after imputation:")
display(df.head())


# In[9]:


categorical_cols = df.select_dtypes(include=['object', 'category']).columns

if len(categorical_cols) == 0:
    print("No categorical columns found. All features are numerical, so no categorical encoding is necessary.")
else:
    print(f"Categorical columns identified: {list(categorical_cols)}")
    # Further steps for encoding would go here if categorical columns were found


# In[10]:


from sklearn.model_selection import train_test_split

# Separate features (X) and target (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data split into training and testing sets successfully.")
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")


# In[11]:


from sklearn.linear_model import LogisticRegression

# Instantiate the Logistic Regression model
log_reg_model = LogisticRegression(random_state=42, solver='liblinear')

# Train the model using the training data
log_reg_model.fit(X_train, y_train)

print("Logistic Regression model initialized and trained successfully.")


# In[12]:


from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Make predictions on the test set
y_pred = log_reg_model.predict(X_test)

# 3. Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 4. Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# 5., 6., 7., 8. Create and display the heatmap of the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[13]:


from sklearn.metrics import roc_curve, roc_auc_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Predict probabilities for the positive class
y_prob = log_reg_model.predict_proba(X_test)[:, 1]

# 2. Calculate ROC curve and ROC-AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = roc_auc_score(y_test, y_prob)

# 3. Print the ROC-AUC score
print(f"ROC-AUC Score: {roc_auc:.2f}")

# 4. Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 5. Make predictions on the test set (reusing y_pred from previous step)
y_pred = log_reg_model.predict(X_test)

# 6. Print the classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# 8. Display the confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', linewidths=.5)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[14]:


import pandas as pd

# Access coefficients and intercept
coefficients = log_reg_model.coef_[0]
intercept = log_reg_model.intercept_[0]

# Create a DataFrame for coefficients
coef_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': coefficients
})

# Add the intercept to the DataFrame
intercept_df = pd.DataFrame([{'Feature': 'Intercept', 'Coefficient': intercept}])
coef_df = pd.concat([coef_df, intercept_df], ignore_index=True)

print("Model Coefficients and Intercept:")
display(coef_df)


# In[ ]:




