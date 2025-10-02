import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from io import BytesIO
import docx
from PyPDF2 import PdfReader
import camelot
import re

st.title("Food Infrastructure Tracker")

# -------------------
# Main Dataset Section
# -------------------
st.header("Main Dataset")
uploaded_file = st.file_uploader("Upload your main dataset (CSV/Excel)", type=["csv", "xlsx"], key="main_dataset")

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.write("### Dataset Preview", df.head())

    # 3D Clustering (needs at least 3 numeric columns)
    numeric_cols = df.select_dtypes(include='number')
    if numeric_cols.shape[1] >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(numeric_cols.iloc[:, :3])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(numeric_cols.iloc[:,0], numeric_cols.iloc[:,1], numeric_cols.iloc[:,2], c=df["Cluster"], cmap="viridis")
        ax.set_xlabel(numeric_cols.columns[0])
        ax.set_ylabel(numeric_cols.columns[1])
        ax.set_zlabel(numeric_cols.columns[2])
        ax.set_title("3D Clustering")
        st.pyplot(fig)
    else:
        st.info("Need at least 3 numeric columns for 3D clustering.")

    # Classification
    try:
        X = df.iloc[:, :-1]
        y = pd.to_numeric(df.iloc[:, -1], errors='coerce')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        tree = DecisionTreeClassifier().fit(X_train, y_train)
        acc = accuracy_score(y_test, tree.predict(X_test))
        st.write("### Decision Tree Accuracy:", round(acc, 2))

        logistic = LogisticRegression(max_iter=500).fit(X_train, y_train)
        preds = logistic.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax2)
        ax2.set_title("Confusion Matrix")
        st.pyplot(fig2)
    except:
        st.info("Target column must be numeric for classification.")

    # Correlation Heatmap
    if numeric_cols.shape[1] >= 2:
        fig3, ax3 = plt.subplots()
        sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", ax=ax3)
        ax3.set_title("Correlation Heatmap")
        st.pyplot(fig3)

# -------------------
# Helper functions for Market Access / Agricultural Yield
# -------------------
def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_pdf(file):
    pdf = PdfReader(file)
    text = ""
    for page in pdf.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_table_from_pdf(file):
    tables = camelot.read_pdf(file, pages='1-end', flavor='lattice')
    if not tables or len(tables) == 0:
        tables = camelot.read_pdf(file, pages='1-end', flavor='stream')
    if tables and len(tables) > 0:
        df = tables[0].df
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(r'[^\d.]','', regex=True), errors='coerce')
        return df
    return None

def extract_table_from_docx(file):
    text = extract_text_from_docx(file)
    try:
        return pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)
    except:
        return None

def extract_table_from_txt(file):
    text = file.read().decode("utf-8")
    try:
        return pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)
    except:
        return None

def extract_table(file):
    if file.name.endswith(".pdf"):
        return extract_table_from_pdf(file)
    elif file.name.endswith(".docx"):
        return extract_table_from_docx(file)
    elif file.name.endswith(".txt"):
        return extract_table_from_txt(file)
    return None

# -------------------
# Market Access Section
# -------------------
st.header("Market Access")
market_file = st.file_uploader("Upload Market Access document (PDF/DOCX/TXT)", key="market")

if market_file:
    if market_file.name.endswith(".pdf"):
        market_text = extract_text_from_pdf(market_file)
    elif market_file.name.endswith(".docx"):
        market_text = extract_text_from_docx(market_file)
    else:
        market_text = market_file.read().decode("utf-8")
    st.subheader("Market Access Content")
    st.text_area("Content", market_text, height=400, key="market_content")

    market_file.seek(0)
    df_market = extract_table(market_file)
    if df_market is not None and not df_market.empty:
        numeric_cols = df_market.select_dtypes(include='number')
        if not numeric_cols.empty:
            st.subheader("Market Access Graph")
            st.bar_chart(numeric_cols)
        else:
            st.info("No numeric data found for graph.")
    else:
        st.info("No table detected. For best results, use CSV/Excel or clear PDF tables.")

# -------------------
# Agricultural Yield Section
# -------------------
st.header("Agricultural Yield")
yield_file = st.file_uploader("Upload Agricultural Yield document (PDF/DOCX/TXT)", key="yield")

if yield_file:
    if yield_file.name.endswith(".pdf"):
        yield_text = extract_text_from_pdf(yield_file)
    elif yield_file.name.endswith(".docx"):
        yield_text = extract_text_from_docx(yield_file)
    else:
        yield_text = yield_file.read().decode("utf-8")
    st.subheader("Agricultural Yield Content")
    st.text_area("Content", yield_text, height=400, key="yield_content")

    yield_file.seek(0)
    df_yield = extract_table(yield_file)
    if df_yield is not None and not df_yield.empty:
        numeric_cols = df_yield.select_dtypes(include='number')
        if not numeric_cols.empty:
            st.subheader("Agricultural Yield Graph")
            st.bar_chart(numeric_cols)
        else:
            st.info("No numeric data found for graph.")
    else:
        st.info("No table detected. For best results, use CSV/Excel or clear PDF tables.")
