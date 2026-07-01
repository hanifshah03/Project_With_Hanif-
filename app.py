import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Page Configuration
st.set_page_config(page_title="Titanic Predictor", page_icon="🚢", layout="centered")
st.title("🚢 Titanic Survival Machine Learning App")
st.write("Adjust the parameters to test your Scikit-Learn Random Forest model live!")

# 2. Automatically load data from a public URL to keep things seamless
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    data = pd.read_csv(url)
    # Basic cleaning to match features
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Age'] = data['Age'].fillna(data['Age'].median())
    return data

df = load_data()

# 3. Train the exact Random Forest Model
features = ["Pclass", "Sex", "Age", "SibSp", "Parch"]
X = df[features]
y = df["Survived"]

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(X, y)

# 4. Build the Web Interface Inputs
st.markdown("### Passenger Demographics")
user_sex = st.selectbox("Select Gender:", ["Male", "Female"])
user_age = st.slider("Select Age:", 1, 100, 25)
user_class = st.selectbox("Ticket Class (1st = Luxury, 3rd = Economy):", [1, 2, 3], index=2)
user_sib = st.slider("Siblings / Spouses Aboard:", 0, 8, 0)
user_parch = st.slider("Parents / Children Aboard:", 0, 6, 0)

# 5. Convert Web UI Choices to model numerical format
sex_numeric = 0 if user_sex == "Male" else 1

# 6. Run Prediction on Button Click
if st.button("Compute Survival Fate"):
    # Structure user input into a tiny DataFrame matching feature names
    user_input = pd.DataFrame([[user_class, sex_numeric, user_age, user_sib, user_parch]], columns=features)
    
    prediction = model.predict(user_input)[0]
    probability = model.predict_proba(user_input)[0][1]
    
    st.markdown("---")
    if prediction == 1:
        st.success(f"🎉 **Survivor!** The ML model predicts a **{probability:.1%}** chance of survival.")
    else:
        st.error(f"💀 **Casualty.** The ML model predicts only a **{probability:.1%}** chance of survival.")
