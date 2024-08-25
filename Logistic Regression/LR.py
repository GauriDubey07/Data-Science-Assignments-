import pandas as pd

# Load the dataset
train_data = pd.read_csv("D:\DS\Logistic Regression\Titanic_train.csv")
test_data = pd.read_csv("D:\DS\Logistic Regression\Titanic_test.csv")

# Display the first few rows of the dataset
train_data.head()






# Check data types and summary statistics
print(train_data.info())
print(train_data.describe())





# Check data types and summary statistics
print(train_data.info())
print(train_data.describe())







from sklearn.impute import SimpleImputer

# Impute missing values for 'Age'
imputer = SimpleImputer(strategy='median')
train_data['Age'] = imputer.fit_transform(train_data[['Age']])
test_data['Age'] = imputer.transform(test_data[['Age']])







from sklearn.preprocessing import LabelEncoder

# Encode 'Sex' column
label_encoder = LabelEncoder()
train_data['Sex'] = label_encoder.fit_transform(train_data['Sex'])
test_data['Sex'] = label_encoder.transform(test_data['Sex'])











from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Define features and target variable
X = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train_data['Survived']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Build and train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)








from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Make predictions
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_prob)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'ROC-AUC Score: {roc_auc}')

# Plot the ROC curve
fpr, tpr, _ = roc_curve(y_val, y_prob)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()







# Interpret the coefficients
coefficients = model.coef_[0]
feature_names = X.columns

for feature, coef in zip(feature_names, coefficients):
    print(f'{feature}: {coef}')







import pickle

# Save the model
with open('logistic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Save the scaler
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)






import streamlit as st
import pickle
import numpy as np

# Load the trained model and preprocessing objects
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Streamlit app
st.title('Titanic Survival Prediction')
st.write('Enter passenger details to predict survival probability.')

# Input fields for user to enter passenger details
pclass = st.selectbox('Pclass', [1, 2, 3])
sex = st.selectbox('Sex', ['male', 'female'])
age = st.slider('Age', 0, 100, 25)
sibsp = st.number_input('SibSp', 0, 10, 0)
parch = st.number_input('Parch', 0, 10, 0)
fare = st.number_input('Fare', 0.0, 500.0, 50.0)

# Encode the 'Sex' input
sex_encoded = label_encoder.transform([sex])[0]

# Prepare the input data
input_data = np.array([[pclass, sex_encoded, age, sibsp, parch, fare]])
input_data = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data)
probability = model.predict_proba(input_data)[0][1]

# Display the prediction
st.write(f'Survival Probability: {probability:.2f}')
st.write(f'Survived: {"Yes" if prediction[0] == 1 else "No"}')

# For online deployment, use Streamlit Community Cloud
# Detailed deployment instructions: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app







import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model and preprocessing objects
with open('logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Streamlit app
st.title('Titanic Survival Prediction')
st.write('Enter passenger details to predict survival probability.')

# Input fields for user to enter passenger details with unique keys
pclass = st.selectbox('Pclass', [1, 2, 3], key='pclass')
sex = st.selectbox('Sex', ['male', 'female'], key='sex')
age = st.slider('Age', 0, 100, 25, key='age')
sibsp = st.number_input('SibSp', 0, 10, 0, key='sibsp')
parch = st.number_input('Parch', 0, 10, 0, key='parch')
fare = st.number_input('Fare', 0.0, 500.0, 50.0, key='fare')

# Encode the 'Sex' input
sex_encoded = label_encoder.transform([sex])[0]

# Prepare the input data
input_data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex_encoded],
    'Age': [age],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare]
})

# Ensure the scaler uses the correct feature names
input_data_scaled = scaler.transform(input_data)

# Make prediction
prediction = model.predict(input_data_scaled)
probability = model.predict_proba(input_data_scaled)[0][1]

# Display the prediction
st.write(f'Survival Probability: {probability:.2f}')
st.write(f'Survived: {"Yes" if prediction[0] == 1 else "No"}')
