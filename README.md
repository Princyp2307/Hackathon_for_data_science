import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the training dataset
train_data_path = '/mnt/data/dataset/loan_sanction_train.csv'
train_data = pd.read_csv(train_data_path)

# Display the first few rows of the dataset
print(train_data.head())

# Check data types and summary statistics
print(train_data.info())
print(train_data.describe())

# Univariate Analysis
sns.histplot(train_data['LoanAmount'])
plt.title('Distribution of Loan Amount')
plt.show()

sns.countplot(x='Gender', data=train_data)
plt.title('Gender Distribution')
plt.show()

sns.countplot(x='Married', data=train_data)
plt.title('Marital Status Distribution')
plt.show()

sns.countplot(x='Education', data=train_data)
plt.title('Education Distribution')
plt.show()

sns.countplot(x='Dependents', data=train_data)
plt.title('Number of Dependents Distribution')
plt.show()

# Bivariate Analysis
sns.scatterplot(x='ApplicantIncome', y='LoanAmount', hue='Loan_Status', data=train_data)
plt.title('Applicant Income vs Loan Amount by Loan Status')
plt.show()

sns.boxplot(x='Education', y='LoanAmount', data=train_data)
plt.title('Loan Amount by Education Level')
plt.show()

# Handle missing values: Fill with median for numerical columns and mode for categorical columns
train_data.fillna(train_data.median(), inplace=True)
train_data.fillna(train_data.mode().iloc[0], inplace=True)

# Feature Engineering: Create new feature 'Income_to_Loan_Ratio'
train_data['Income_to_Loan_Ratio'] = train_data['ApplicantIncome'] / train_data['LoanAmount']

# Define features and target variable
X = train_data.drop('Loan_Status', axis=1)
y = train_data['Loan_Status']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

# Predict and evaluate Logistic Regression model
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, lr_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_predictions))

# Train Random Forest model
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

# Predict and evaluate Random Forest model
rf_predictions = rf_model.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))

# Re-train models after feature engineering
X = train_data.drop('Loan_Status', axis=1)
y = train_data['Loan_Status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Re-train Logistic Regression model with new feature
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
print("Logistic Regression with FE Accuracy:", accuracy_score(y_test, lr_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, lr_predictions))

# Re-train Random Forest model with new feature
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
print("Random Forest with FE Accuracy:", accuracy_score(y_test, rf_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_predictions))
