# Save as app.py
# Run: streamlit run app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import docx
from PyPDF2 import PdfReader

st.title("Food Infrastructure Tracker")

# -------------------
# Upload Dataset (CSV/Excel)
# -------------------
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Load dataset
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Preview of Uploaded Data", df.head())

    # Assume last column is Target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(X)

    fig1 = plt.figure()
    ax = fig1.add_subplot(111, projection="3d")
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:, 2], c=df["Cluster"], cmap="viridis")
    ax.set_xlabel(X.columns[0]); ax.set_ylabel(X.columns[1]); ax.set_zlabel(X.columns[2])
    ax.set_title("3D Clustering Result")
    st.pyplot(fig1)

    # Train/test for supervised models
    labels = ["Low", "Medium", "High"]
    try:
        y_bins = pd.qcut(y, q=len(labels), labels=labels)
    except Exception:
        y_bins = pd.cut(y, bins=len(labels), labels=labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y_bins, test_size=0.3, random_state=42)

    # Decision Tree
    tree = DecisionTreeClassifier().fit(X_train, y_train)
    acc = accuracy_score(y_test, tree.predict(X_test))
    st.write("### Decision Tree Accuracy:", round(acc, 2))

    # Logistic Regression + confusion matrix
    logistic = LogisticRegression(max_iter=500).fit(X_train, y_train.cat.codes)
    preds = logistic.predict(X_test)
    cm = confusion_matrix(y_test.cat.codes, preds)

    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax2)
    ax2.set_title("Confusion Matrix")
    st.pyplot(fig2)

    # Correlation Heatmap
    fig3, ax3 = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax3)
    ax3.set_title("Correlation Heatmap (Feature Relationships)")
    st.pyplot(fig3)

# -------------------
# Upload Document (TXT, DOCX, PDF) for Market Access / Yield
# -------------------
st.header("Upload External Document for Market/Yield Analysis")
doc_file = st.file_uploader("Upload a document (TXT, DOCX, PDF)", type=["txt", "docx", "pdf"])

if doc_file:
    text = ""

    if doc_file.name.endswith(".txt"):
        text = doc_file.read().decode("utf-8")
    elif doc_file.name.endswith(".docx"):
        doc = docx.Document(doc_file)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif doc_file.name.endswith(".pdf"):
        pdf = PdfReader(doc_file)
        for page in pdf.pages:
            text += page.extract_text()

    st.write("### Extracted Text from Document:")
    st.text_area("Preview", text[:1000], height=200)

    # Example: Keyword-based analysis
    keywords = ["market", "access", "agriculture", "yield", "production", "infrastructure"]
    counts = {k: text.lower().count(k) for k in keywords}

    # Plot graph
    fig4, ax4 = plt.subplots()
    ax4.bar(counts.keys(), counts.values(), color="green")
    ax4.set_title("Keyword Frequency in Document (Market Access / Agricultural Yield)")
    ax4.set_ylabel("Count")
    st.pyplot(fig4)
