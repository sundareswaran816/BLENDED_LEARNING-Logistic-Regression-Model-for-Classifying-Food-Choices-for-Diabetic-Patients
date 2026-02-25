# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection and Preprocessing Load the food nutrition dataset, handle missing values, encode categorical labels, and normalize the nutritional features using scaling techniques.

2. Data Splitting Split the dataset into training and testing sets using an appropriate ratio (e.g., 80% training and 20% testing).

3. Model Training Train a Logistic Regression model using the training dataset to learn the relationship between nutritional features and diabetic suitability.

4. Model Evaluation and Prediction Test the model on the testing dataset and evaluate performance using accuracy, precision, recall, F1-score, and confusion matrix. Use the trained model to classify food items as suitable or unsuitable for diabetic patients.

## Program:
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

#load dataset 
df=pd.read_csv("food_items (1).csv")
#inspect the dataset
print("Dataset Overview")
print(df.head())
print("\ndatset Info")
print(df.info())

X_raw=df.iloc[:, :-1]
y_raw=df.iloc[:, -1:]
X_raw

scaler=MinMaxScaler()
X=scaler.fit_transform(X_raw)

label_encoder=LabelEncoder()
y=label_encoder.fit_transform(y_raw.values.ravel())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,stratify=y,random_state=123)

penalty='l2'
multi_class='multnomial'
solver='lbfgs'
max_iter=1000

model = LogisticRegression(max_iter=2000)  # Increased max_iter for convergence
model.fit(X_train, y_train)

# Model Prediction
y_pred = model.predict(X_test)

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='coolwarm', cbar=False, 
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
```

## Output:
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/743216a5-df57-4d7a-b9b9-682e3820c43f" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/a1f42135-dbe6-4a1e-8f6d-a38d3c40c842" />
<img width="1920" height="1020" alt="image" src="https://github.com/user-attachments/assets/1b3c502a-5e77-491e-b60d-c7fc4c6dda73" />



## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
