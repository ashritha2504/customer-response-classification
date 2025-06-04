# #<========================>
# ## 2. GET THE DATA
# #<========================>

import pandas as pd

# # # Load the dataset
file_path = "bank-additional-full.csv"
data = pd.read_csv(file_path, sep=';')

# Verify the dataset is loaded correctly
print("Dataset successfully loaded.")
print("\n--- First 5 rows of the dataset ---")
print(data.head())

print("\n--- Dataset Information ---")
print(data.info())

print("\n--- Descriptive Statistics for Numerical Features ---")
print(data.describe())

# Display the shape of the dataset
print("\n--- Dataset Shape ---")
print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Examine missing or unknown values
print("\n--- Missing Values ---")
missing_counts = (data == "unknown").sum()
print(missing_counts[missing_counts > 0])


# #<========================>
# ## 3.	Explore and visualize the data to gain insights. 
# #<========================>

import matplotlib.pyplot as plt
import seaborn as sns


# Bar chart for categorical variables (job)
plt.figure(figsize=(10, 6))
data['job'].value_counts().plot(kind='bar', color='skyblue')
plt.title('Job Distribution')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram for numerical variables (age)
plt.figure(figsize=(8, 6))
data['age'].plot(kind='hist', bins=20, color='lightgreen', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Box plot for duration
plt.figure(figsize=(8, 6))
sns.boxplot(data=data, y='duration', color='salmon')
plt.title('Box Plot of Call Duration')
plt.ylabel('Duration (seconds)')
plt.tight_layout()
plt.show()

# Correlation heatmap
numerical_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 
                  'cons.conf.idx', 'euribor3m', 'nr.employed']
plt.figure(figsize=(12, 10))
correlation_matrix = data[numerical_cols].corr()
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# Statistical summary
print("\n--- Statistical Summary ---")
print(data.describe())

# Frequency of target variable
print("\n--- Target Variable Distribution ---")
print(data['y'].value_counts(normalize=True) * 100)




# #<========================>
# ## 4.	Prepare the data for machine learning algorithms
# #<========================>

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# Separate features and target variable
X = data.drop(columns=['y'])
y = data['y']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Handling missing values
categorical_imputer = SimpleImputer(strategy='most_frequent')
numerical_imputer = SimpleImputer(strategy='mean')

# Encoding categorical variables and scaling numerical variables
categorical_transformer = Pipeline(steps=[
    ('imputer', categorical_imputer),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', numerical_imputer),
    ('scaler', StandardScaler())
])

# Combine preprocessors into a column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.3, stratify=y, random_state=42)

# Verify results
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# export the tidy partitioned datasets

# Convert training and testing datasets to DataFrames
X_train_df = pd.DataFrame(X_train)
X_test_df = pd.DataFrame(X_test)
y_train_df = pd.DataFrame(y_train, columns=['target'])  # Assuming 'y' is the target variable
y_test_df = pd.DataFrame(y_test, columns=['target'])

# Combine features and target for training and testing datasets
train_df = pd.concat([X_train_df, y_train_df], axis=1)
test_df = pd.concat([X_test_df, y_test_df], axis=1)

# Export to CSV files
train_df.to_csv('train_dataset.csv', index=False)
test_df.to_csv('test_dataset.csv', index=False)

print("Training and testing datasets have been exported as 'train_dataset.csv' and 'test_dataset.csv'")




# # #<========================>
# # ## 5.	Select a model and train it. 
# # #<========================>

import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y, test_size=0.3, stratify=y, random_state=42
)

# Initialize logistic regression model with regularization
log_reg = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)

# Train the model
log_reg.fit(X_train, y_train)

# Evaluate performance
y_pred = log_reg.predict(X_test)
y_proba = log_reg.predict_proba(X_test)[:, 1]

# Print classification report
# print("Classification Report:\n", classification_report(y_test, y_pred))

# Generate learning curves
train_sizes = np.linspace(0.1, 0.9999, 5)  # Avoid reaching exactly 1.0
train_scores, valid_scores = [], []

for train_size in train_sizes:
    # Dynamically ensure train_size remains valid
    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=float(train_size), random_state=42)
    log_reg.fit(X_sub, y_sub)
    train_scores.append(log_reg.score(X_sub, y_sub))
    valid_scores.append(log_reg.score(X_test, y_test))

# Plot learning curve
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label='Training Score')
plt.plot(train_sizes, valid_scores, label='Validation Score')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Learning Curve')
plt.legend()
plt.grid()
plt.show()



#<========================>
## 6.	Fine tune your model.
#<========================>

from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score


# Encode target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # "no" -> 0, "yes" -> 1

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X_preprocessed, y_encoded, test_size=0.3, stratify=y_encoded, random_state=42
)

# Logistic Regression model
log_reg = LogisticRegression(penalty='l2', solver='liblinear', random_state=42)

# Hyperparameter tuning
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(
    log_reg,
    param_grid,
    scoring=make_scorer(f1_score, pos_label=1),  # Explicitly set pos_label
    cv=5,
    verbose=1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Best parameters and the fine-tuned model
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Evaluate on the test set
y_pred_best = best_model.predict(X_test)
print("Fine-Tuned Classification Report:\n", classification_report(y_test, y_pred_best, target_names=label_encoder.classes_))

# Plot updated learning curve
train_sizes = np.linspace(0.1, 0.9999, 5)
train_scores, valid_scores = [], []

for train_size in train_sizes:
    X_sub, _, y_sub, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)
    best_model.fit(X_sub, y_sub)
    train_scores.append(best_model.score(X_sub, y_sub))
    valid_scores.append(best_model.score(X_test, y_test))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, label='Training Score (Fine-Tuned)')
plt.plot(train_sizes, valid_scores, label='Validation Score (Fine-Tuned)')
plt.xlabel('Training Set Size')
plt.ylabel('Score')
plt.title('Fine-Tuned Learning Curve')
plt.legend()
plt.grid()
plt.show()
