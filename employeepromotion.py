# %%
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from joblib import dump, load
import lime.lime_tabular
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# %%
FileLocation = os.getcwd()

print("The File Location: ", os.getcwd())

# %%
EmployeePromotion = pd.read_csv('train.csv')
EmployeePromotion.head()

# %%
EmployeePromotion.shape

# %%
EmployeePromotion.describe()

# %%
EmployeePromotion.isnull().sum()

# %%
EmployeePromotion.tail()

# %%
EmployeePromotion.dtypes

# %%
EmployeePromotion["education"].fillna(value=EmployeePromotion["education"].mode()[0], inplace=True)

# %%
EmployeePromotion["previous_year_rating"].fillna(value=EmployeePromotion["previous_year_rating"].median(), inplace=True)

# %%
EmployeePromotion.isnull().sum()

# %%
EmployeePromotion.describe(include=['object'])

# %%
EmployeePromotion['is_promoted'].value_counts()

# %%
fig, ax = plt.subplots(2, 3, figsize=(15, 10))
sns.countplot(x="gender", data=EmployeePromotion, ax=ax[0,0])
sns.countplot(x="department", data=EmployeePromotion, ax=ax[0,1])
ax[0,1].set_xticklabels(ax[0,1].get_xticklabels(), rotation=45, ha='right')
sns.countplot(x="no_of_trainings", data=EmployeePromotion, ax=ax[0,2])
sns.countplot(x="education", data=EmployeePromotion, ax=ax[1,0])
sns.countplot(x="awards_won?", data=EmployeePromotion, ax=ax[1,1])
sns.countplot(x="is_promoted", data=EmployeePromotion, ax=ax[1,2])
fig.tight_layout()  
fig.show()

# %%
fig, ax = plt.subplots(2,3, figsize=(15, 10))
sns.countplot(x="gender", hue="department", data=EmployeePromotion, ax=ax[0,0])
sns.countplot(x="gender", hue="no_of_trainings", data=EmployeePromotion, ax=ax[0,1])
sns.countplot(x="gender", hue="education", data=EmployeePromotion, ax=ax[0,2])
sns.countplot(x="gender", hue="recruitment_channel", data=EmployeePromotion, ax=ax[1,0])
sns.countplot(x="gender", hue="awards_won?", data=EmployeePromotion, ax=ax[1,1])
sns.countplot(x="gender", hue="is_promoted", data=EmployeePromotion, ax=ax[1,2])
fig.show()

# %%
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
sns.countplot(x="gender", hue="region", data=EmployeePromotion, ax=ax)
fig.show()

# %%
EmployeePromotion.dtypes

# %%
# Create a 2x3 subplot grid
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

# Plot histograms for each numerical feature
sns.histplot(x="age", data=EmployeePromotion, ax=ax[0, 0])
sns.histplot(x="previous_year_rating", data=EmployeePromotion, ax=ax[0, 1])
sns.histplot(x="length_of_service", data=EmployeePromotion, ax=ax[1, 0])
sns.histplot(x="avg_training_score", data=EmployeePromotion, ax=ax[1, 1])

# Display the figure
fig.show()

# %%
EmployeePromotion.describe()

# %%
# Bin numerical features into categories for use as hue
EmployeePromotion['age_bin'] = pd.cut(EmployeePromotion['age'], bins=[19, 30, 40, 60], labels=['Young', 'Middle', 'Senior'])
EmployeePromotion['prev_rating_bin'] = pd.cut(EmployeePromotion['previous_year_rating'], bins=[0, 2, 4, 5], labels=['Low', 'Medium', 'High'])
EmployeePromotion['service_bin'] = pd.cut(EmployeePromotion['length_of_service'], bins=[0, 5, 10, 40], labels=['Short', 'Medium', 'Long'])
EmployeePromotion['training_score_bin'] = pd.cut(EmployeePromotion['avg_training_score'], bins=[0, 60, 80, 100], labels=['Low', 'Medium', 'High'])
EmployeePromotion['trainings_bin'] = pd.cut(EmployeePromotion['no_of_trainings'], bins=[0, 1, 3, 10], labels=['Few', 'Some', 'Many'])
EmployeePromotion['awards_bin'] = pd.cut(EmployeePromotion['awards_won?'], bins=[-1, 0, 1], labels=['No', 'Yes'])  # Binary-like

# Create a 2x3 subplot grid
fig, ax = plt.subplots(2, 3, figsize=(15, 10))

# Plot count plots with gender as x and binned numerical features as hue
sns.countplot(x="gender", hue="age_bin", data=EmployeePromotion, ax=ax[0, 0])
sns.countplot(x="gender", hue="prev_rating_bin", data=EmployeePromotion, ax=ax[0, 1])
sns.countplot(x="gender", hue="service_bin", data=EmployeePromotion, ax=ax[0, 2])
sns.countplot(x="gender", hue="training_score_bin", data=EmployeePromotion, ax=ax[1, 0])
sns.countplot(x="gender", hue="trainings_bin", data=EmployeePromotion, ax=ax[1, 1])
sns.countplot(x="gender", hue="awards_bin", data=EmployeePromotion, ax=ax[1, 2])

# Display the figure
fig.show()

# %%
# Create a 4x3 subplot grid
fig, ax = plt.subplots(4, 3, figsize=(15, 15))

# Plot countplots in the specified subplots
sns.countplot(x="gender", hue="is_promoted", data=EmployeePromotion, ax=ax[0, 0])
sns.countplot(x="department", hue="is_promoted", data=EmployeePromotion, ax=ax[0, 1])
ax[0, 1].set_xticklabels(ax[0, 1].get_xticklabels(), rotation=45, ha='right')
sns.countplot(x="no_of_trainings", hue="is_promoted", data=EmployeePromotion, ax=ax[0, 2])
sns.countplot(x="education", hue="is_promoted", data=EmployeePromotion, ax=ax[1, 0])
sns.countplot(x="recruitment_channel", hue="is_promoted", data=EmployeePromotion, ax=ax[1, 1])
sns.countplot(x="awards_won?", hue="is_promoted", data=EmployeePromotion, ax=ax[1, 2])
sns.countplot(x="age_bin", hue="is_promoted", data=EmployeePromotion, ax=ax[2, 0])
sns.countplot(x="prev_rating_bin", hue="is_promoted", data=EmployeePromotion, ax=ax[2, 1])
sns.countplot(x="service_bin", hue="is_promoted", data=EmployeePromotion, ax=ax[2, 2])
sns.countplot(x="training_score_bin", hue="is_promoted", data=EmployeePromotion, ax=ax[3, 0])
sns.countplot(x="trainings_bin", hue="is_promoted", data=EmployeePromotion, ax=ax[3, 1])

# Remove the empty subplot at ax[3, 2] to make it not show
ax[3, 2].remove()

# Adjust layout to prevent label overlap
fig.tight_layout()

# Display the figure
plt.show()

# %%
EmployeePromotion.dtypes

# %%
# 1. Check for duplicate rows (all columns identical)
duplicate_count = EmployeePromotion.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# 2. Display duplicate rows (if any)
duplicates = EmployeePromotion[EmployeePromotion.duplicated(keep=False)]
if not duplicates.empty:
    print("\nDuplicate rows:")
    print(duplicates)
else:
    print("\nNo duplicate rows found.")

# %%
EmployeePromotion = EmployeePromotion.drop(columns=['employee_id', 'age_bin', 'prev_rating_bin', 'service_bin', 'training_score_bin', 'trainings_bin', 'awards_bin'],axis=1)
EmployeePromotion.head()

# %%
# Print unique values for categorical features (object dtype)
categorical_cols = EmployeePromotion.select_dtypes(include=['object']).columns
print("\nUnique Values for Categorical Features (object dtype):")
for col in categorical_cols:
    unique_vals = EmployeePromotion[col].unique()
    print(f"{col}: {unique_vals} (Count: {len(unique_vals)})")

# %%
EmployeePromotion["gender"].replace({'m': 1, 'f': 0}, inplace=True)
EmployeePromotion["education"].replace({"Below Secondary": 0, "Bachelor's": 1, "Master's & above": 2}, inplace=True)
EmployeePromotion.head()

# %%
# Assuming EmployeePromotion is your DataFrame

# List of columns to label encode
EmployeePromotion_Label = ['department', 'region', 'recruitment_channel']

# Create a LabelEncoder instance
label_encoder = LabelEncoder()

# Create a copy of the DataFrame to avoid modifying the original
EmployeePromotion_encoded = EmployeePromotion.copy()

# Apply label encoding to each specified column
for column in EmployeePromotion_Label:
    # Fit and transform the column, updating the DataFrame
    EmployeePromotion_encoded[column] = label_encoder.fit_transform(EmployeePromotion_encoded[column])
    
    # Optional: Print the mapping for reference
    print(f"\nLabel Encoding for '{column}':")
    unique_values = EmployeePromotion[column].unique()
    encoded_values = EmployeePromotion_encoded[column].unique()
    mapping = dict(zip(unique_values, encoded_values))
    print(mapping)

# Verify the result
print("\nData Types after encoding:\n", EmployeePromotion_encoded.dtypes)
print("\nFirst few rows of the encoded DataFrame:\n", EmployeePromotion_encoded.head())

# %%
# Calculate the correlation matrix
corr_matrix = EmployeePromotion_encoded.corr()

# Plot heatmap with annotations formatted to 1 decimal place
sns.heatmap(corr_matrix, annot=True, fmt=".1f")

# Display the plot
plt.show()

# %%
# Create the attribute and target data:
X=EmployeePromotion_encoded.drop(columns=['is_promoted'],axis=1)
y=EmployeePromotion_encoded['is_promoted']
print(X.head())
print()
print(y.head())

# %%
# Split the data into training and test data using train_test_split() function. :
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, stratify=y, random_state=2)
print(X.shape,X_train.shape,X_test.shape)

# %%
# Initialize the Gradient Boosting model with default parameters
model = GradientBoostingClassifier()

# Fit the model to the training data
model.fit(X_train, y_train)

# %%
# Get feature importance
feature_importance = model.feature_importances_

# Create a DataFrame to display feature names and their importance scores
feature_names = X_train.columns  # Assuming X_train is a pandas DataFrame
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
})

# Sort the DataFrame by importance in descending order
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the feature importance
print("Feature Importance:")
print(importance_df)

# Visualize the feature importance using a bar plot
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')

# %%
dump(model,"GBC_EmployeePromotion.joblib")

# %%
explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
feature_names=X_train.columns,
verbose=True, mode='classification')

# %%
explanation = explainer.explain_instance(X_test.iloc[1], model.predict_proba, num_features=11)

# %%
from IPython.display import display, HTML
# Force light background with custom styling if dark theme is the issue
html_content = explanation.as_html(show_table=True)
styled_html = f"""
<div style='background-color: white; color: black; padding: 10px;'>
    {html_content}
</div>
"""
display(HTML(styled_html))

# %%
explanation.as_list()

# %%
# Evaluate using the training data
train_predict=model.predict(X_train)
print("Accuracy on training data: ", metrics.accuracy_score(y_train, train_predict))
print("Precision on training data:", metrics.precision_score(y_train, train_predict))
print("Recall on training data:", metrics.recall_score(y_train, train_predict))

# %%
# Evaluate using the testing data
test_predict=model.predict(X_test)
print("Accuracy on testing data: ", metrics.accuracy_score(y_test, test_predict))
print("Precision on testing data:", metrics.precision_score(y_test, test_predict))
print("Recall on testing data: ", metrics.recall_score(y_test, test_predict))

# %%
cm = metrics.confusion_matrix(y_test, test_predict)
TN, FP, FN, TP = cm.ravel()
print("TN={0}, FP={1}, FN={2}, TP={3}".format(TN, FP, FN, TP))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format='d')
plt.show()

# %%
calculated_accuracy = (TP+TN)/(TP+TN+FP+FN)
calculated_precision = (TP)/(TP+FP)
calculated_recall = (TP)/(TP+FN)
calculated_false_positive_rate = (FP)/(FP+TN)
print("Calculated accuracy = ", calculated_accuracy)
print("Calculated precision = ", calculated_precision)
print("Calculated recall = ", calculated_recall)
print("Calculated false positive rate = ", calculated_false_positive_rate)


# %%
PROTECTED  =  "gender" 
MALE =  1.0 
FEMALE = 0.0 
gender_dist = X_test[PROTECTED].value_counts()
print(gender_dist)
male_indices = np.where(X_test[PROTECTED] == MALE)[0]
female_indices = np.where(X_test[PROTECTED] == FEMALE)[0]
print(male_indices, "No of Male =", male_indices.size)
print(female_indices, "No of Female =", female_indices.size)

# %%
print(y_test)

# %%
print(y_test[:20])

# %%
explanation = explainer.explain_instance(X_test.iloc[7], model.predict_proba)

# %%
from IPython.display import display, HTML
# Force light background with custom styling if dark theme is the issue
html_content = explanation.as_html(show_table=True)
styled_html = f"""
<div style='background-color: white; color: black; padding: 10px;'>
    {html_content}
</div>
"""
display(HTML(styled_html))

# %%
explanation.as_list()

# %%
y_test_m = [y_test.values[i] for i in male_indices]
y_test_f = [y_test.values[i] for i in female_indices]
print(y_test_m)
print(y_test_f)

# %%
y_predict_m = [test_predict[i] for i in male_indices]
y_predict_f = [test_predict[i] for i in female_indices]
print(y_predict_m)
print()
print(y_predict_f)

# %%
cm_m = metrics.confusion_matrix(y_test_m, y_predict_m)
TN, FP, FN, TP = cm_m.ravel()
print("TN={0}, FP={1}, FN={2}, TP={3}".format(TN, FP, FN, TP))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_m)
disp.plot()
plt.show()

# %%
calculated_accuracy = (TP+TN)/(TP+TN+FP+FN)
calculated_precision = (TP)/(TP+FP)
calculated_recall = (TP)/(TP+FN)
calculated_positive_rate = (TP+FP)/(TP+TN+FP+FN)
print("Calculated accuracy = ", calculated_accuracy)
print("Calculated precision = ", calculated_precision)
print("Calculated recall = ", calculated_recall)
print("Calculated positive rate = ", calculated_positive_rate)

# %%
cm_f = metrics.confusion_matrix(y_test_f, y_predict_f)
TN, FP, FN, TP = cm_f.ravel()
print("TN={0}, FP={1}, FN={2}, TP={3}".format(TN, FP, FN, TP))
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_f)
disp.plot()
plt.show()

# %%
calculated_accuracy = (TP+TN)/(TP+TN+FP+FN)
calculated_precision = (TP)/(TP+FP)
calculated_recall = (TP)/(TP+FN)
calculated_positive_rate = (TP+FP)/(TP+TN+FP+FN)
print("Calculated accuracy = ", calculated_accuracy)
print("Calculated precision = ", calculated_precision)
print("Calculated recall = ", calculated_recall)
print("Calculated positive rate = ", calculated_positive_rate)


