import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load("phishing_URL_detector/model.pkl")
df=pd.read_csv("phishing_URL_detector/dataset.csv")
df.drop('index',axis=1,inplace=True)
X=df.drop('Result',axis=1)
y=df['Result']
X_train ,X_test ,y_train ,y_test =train_test_split(X ,y ,test_size=0.2 ,stratify=y ,random_state=42)
y_pred=model.predict(X_test)
st.title("Phishing Website Detector")
st.markdown("#### By supervision of Dr. Mahmoud Yasin")

st.subheader("Our model performance")
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
st.pyplot(fig)
st.subheader("Classification Report")
st.dataframe(df_report)

mode = st.radio("Choose Input Mode:", ["Manual Input", "Upload File"])

if mode == "Manual Input":

    st.subheader("Enter Features Manually")

    having_IP = st.number_input("having_IP_Address", min_value=-1, max_value=1, value=0)
    url_length = st.number_input("URL_Length", min_value=-1, max_value=1, value=0)
    shortening_service = st.number_input("Shortening_Service", min_value=-1, max_value=1, value=0)
    having_at_symbol = st.number_input("having_At_Symbol", min_value=-1, max_value=1, value=0)
    double_slash = st.number_input("double_slash_redirecting", min_value=-1, max_value=1, value=0)
    prefix_suffix = st.number_input("Prefix_Suffix", min_value=-1, max_value=1, value=0)
    sub_domain = st.number_input("having_Sub_Domain", min_value=-1, max_value=1, value=0)
    ssl_final_state = st.number_input("SSLfinal_State", min_value=-1, max_value=1, value=0)
    domain_reg = st.number_input("Domain_registeration_length", min_value=-1, max_value=1, value=0)
    favicon = st.number_input("Favicon", min_value=-1, max_value=1, value=0)

    port = st.number_input("port", min_value=-1, max_value=1, value=0)
    https_token = st.number_input("HTTPS_token", min_value=-1, max_value=1, value=0)
    request_url = st.number_input("Request_URL", min_value=-1, max_value=1, value=0)
    url_anchor = st.number_input("URL_of_Anchor", min_value=-1, max_value=1, value=0)
    links_tags = st.number_input("Links_in_tags", min_value=-1, max_value=1, value=0)
    sfh = st.number_input("SFH", min_value=-1, max_value=1, value=0)
    email_submit = st.number_input("Submitting_to_email", min_value=-1, max_value=1, value=0)
    abnormal_url = st.number_input("Abnormal_URL", min_value=-1, max_value=1, value=0)
    redirect = st.number_input("Redirect", min_value=-1, max_value=1, value=0)
    mouseover = st.number_input("on_mouseover", min_value=-1, max_value=1, value=0)
    right_click = st.number_input("RightClick", min_value=-1, max_value=1, value=0)
    popup = st.number_input("popUpWidnow", min_value=-1, max_value=1, value=0)
    iframe = st.number_input("Iframe", min_value=-1, max_value=1, value=0)
    age_domain = st.number_input("age_of_domain", min_value=-1, max_value=1, value=0)
    dns = st.number_input("DNSRecord", min_value=-1, max_value=1, value=0)
    web_traffic = st.number_input("web_traffic", min_value=-1, max_value=1, value=0)
    page_rank = st.number_input("Page_Rank", min_value=-1, max_value=1, value=0)
    google_index = st.number_input("Google_Index", min_value=-1, max_value=1, value=0)
    links_pointing = st.number_input("Links_pointing_to_page", min_value=-1, max_value=1, value=0)
    stat_report = st.number_input("Statistical_report", min_value=-1, max_value=1, value=0)

    if st.button("Predict"):

        input_data = np.array([[
            having_IP, url_length, shortening_service, having_at_symbol,
            double_slash, prefix_suffix, sub_domain, ssl_final_state,
            domain_reg, favicon, port, https_token, request_url,
            url_anchor, links_tags, sfh, email_submit, abnormal_url,
            redirect, mouseover, right_click, popup, iframe,
            age_domain, dns, web_traffic, page_rank, google_index,
            links_pointing, stat_report
        ]])

        pred = model.predict(input_data)[0]

        if pred == 1:
            st.success("✅ Legitimate Website")
        else:
            st.error("⚠️ Phishing Website")


else:

    st.subheader("Upload CSV / Excel / JSON File")

    file = st.file_uploader("Upload file", type=["csv", "xlsx", "json"])

    if file is not None:

        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        else:
            df = pd.read_json(file)

        st.write("Preview:")
        st.dataframe(df.head())

        preds = model.predict(df)

        df["Prediction"] = preds
        df["Prediction"] = df["Prediction"].map({
            1: "Legitimate",
            -1: "Phishing"
        })

        st.subheader("Results")
        st.dataframe(df)

        # stats
        st.write("Legitimate:", (df["Prediction"] == "Legitimate").sum())
        st.write("Phishing:", (df["Prediction"] == "Phishing").sum())
