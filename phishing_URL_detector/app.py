import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


model = joblib.load("phishing_URL_detector/model.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("phishing_URL_detector/dataset.csv")
    df = df.drop(columns=["index"], errors="ignore")

    X = df.drop("Result", axis=1)
    y = df["Result"]

    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

X_train, X_test, y_train, y_test = load_data()


y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()


st.title("Phishing Website Detector")
st.caption("Supervised by Dr. Mahmoud Yasin")

st.subheader("Our model performance")


st.subheader("Classification Report")
st.dataframe(df_report, use_container_width=True)


fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='copper', ax=ax)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

st.pyplot(fig)


mode = st.radio("Choose Input Mode:", ["Manual Input", "Upload File"])


if mode == "Manual Input":

    st.subheader("Enter Features")

    inputs = []

    features_names = [
        "having_IP_Address","URL_Length","Shortining_Service","having_At_Symbol",
        "double_slash_redirecting","Prefix_Suffix","having_Sub_Domain","SSLfinal_State",
        "Domain_registeration_length","Favicon","port","HTTPS_token","Request_URL",
        "URL_of_Anchor","Links_in_tags","SFH","Submitting_to_email","Abnormal_URL",
        "Redirect","on_mouseover","RightClick","popUpWidnow","Iframe",
        "age_of_domain","DNSRecord","web_traffic","Page_Rank","Google_Index",
        "Links_pointing_to_page","Statistical_report"
    ]

    for f in features_names:
        val = st.number_input(f, min_value=-1, max_value=1, value=0)
        inputs.append(val)

    if st.button("Predict"):
        input_data = np.array(inputs).reshape(1, -1)
        pred = model.predict(input_data)[0]

        if pred == 1:
            st.success("Legitimate Website :D")
        else:
            st.error("Phishing Website :'(")


else:

    st.subheader("Upload CSV / Excel / JSON")

    file = st.file_uploader("Upload file", type=["csv", "xlsx", "json"])

    if file is not None:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_json(file)

        
        df = df.drop(columns=["index", "Result"], errors="ignore")

        st.write("Preview:")
        st.dataframe(df.head())

        try:
            preds = model.predict(df)

            df["Prediction"] = preds
            df["Prediction"] = df["Prediction"].map({
                1: "Legitimate",
                -1: "Phishing"
            })

            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

            st.write("Legitimate:", (df["Prediction"] == "Legitimate").sum())
            st.write("Phishing:", (df["Prediction"] == "Phishing").sum())

        except Exception as e:
            st.error(f"Error: {e}")
