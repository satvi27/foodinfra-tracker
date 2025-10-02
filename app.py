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
from io import StringIO

st.title("Food Infrastructure Tracker")

# -------------------
# Upload Dataset (CSV/Excel)
# -------------------
st.header("Upload Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

if uploaded_file:
    try:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    st.write("### Preview of Uploaded Data", df.head())

    X_numeric = df.select_dtypes(include=np.number).fillna(0)
    y = df.iloc[:, -1]

    # 3D Clustering
    if X_numeric.shape[1] >= 3:
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        df["Cluster"] = kmeans.fit_predict(X_numeric)

        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection="3d")
        ax.scatter(
            X_numeric.iloc[:, 0],
            X_numeric.iloc[:, 1],
            X_numeric.iloc[:, 2],
            c=df["Cluster"],
            cmap="viridis"
        )
        ax.set_xlabel(X_numeric.columns[0])
        ax.set_ylabel(X_numeric.columns[1])
        ax.set_zlabel(X_numeric.columns[2])
        ax.set_title("3D Clustering Result")
        st.pyplot(fig1)
    else:
        st.warning("Dataset must have at least 3 numeric columns for 3D clustering.")

    # Classification
    if pd.api.types.is_numeric_dtype(y) and X_numeric.shape[1] > 0:
        labels = ["Low", "Medium", "High"]
        try:
            y_bins = pd.qcut(y, q=len(labels), labels=labels)
        except Exception:
            y_bins = pd.cut(y, bins=len(labels), labels=labels)

        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y_bins, test_size=0.3, random_state=42)

        tree = DecisionTreeClassifier().fit(X_train, y_train)
        acc = accuracy_score(y_test, tree.predict(X_test))
        st.write("### Decision Tree Accuracy:", round(acc, 2))

        logistic = LogisticRegression(max_iter=500).fit(X_train, y_train.cat.codes)
        preds = logistic.predict(X_test)
        cm = confusion_matrix(y_test.cat.codes, preds)

        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax2)
        ax2.set_title("Confusion Matrix")
        st.pyplot(fig2)
    else:
        st.warning("Target column must be numeric for classification.")

    # Correlation Heatmap
    if X_numeric.shape[1] > 0:
        fig3, ax3 = plt.subplots()
        sns.heatmap(X_numeric.corr(), annot=True, cmap="coolwarm", ax=ax3)
        ax3.set_title("Correlation Heatmap (Numeric Features)")
        st.pyplot(fig3)
    else:
        st.warning("No numeric columns available for correlation heatmap.")

# -------------------
# Helper functions to extract text and table
# -------------------
def extract_text(file):
    text = ""
    if file.name.endswith(".txt"):
        text = file.read().decode("utf-8")
    elif file.name.endswith(".docx"):
        doc = docx.Document(file)
        text = "\n".join([p.text for p in doc.paragraphs])
    elif file.name.endswith(".pdf"):
        pdf = PdfReader(file)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

def extract_table_from_text(text):
    """
    Convert tabular text to DataFrame if possible
    """
    try:
        df_table = pd.read_csv(StringIO(text), delim_whitespace=True)
        return df_table
    except Exception:
        return None

# -------------------
# Market Access Section
# -------------------
st.header("Market Access")
market_file = st.file_uploader("Upload Market Access document", type=["txt", "docx", "pdf"], key="market")

if market_file:
    market_text = extract_text(market_file)
    st.subheader("Market Access Details from Document")
    st.text_area("Market Access Content", market_text, height=400)

    # Try to extract table and plot
    df_market = extract_table_from_text(market_text)
    if df_market is not None:
        numeric_cols = df_market.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            st.subheader("Market Access Graphs")
            st.bar_chart(numeric_cols)
        else:
            st.info("No numeric data found in Market Access document for graph.")
    else:
        st.info("Could not detect tabular data in Market Access document.")

# -------------------
# Agricultural Yield Section
# -------------------
st.header("Agricultural Yield")
yield_file = st.file_uploader("Upload Agricultural Yield document", type=["txt", "docx", "pdf"], key="yield")

if yield_file:
    yield_text = extract_text(yield_file)
    st.subheader("Agricultural Yield Details from Document")
    st.text_area("Agricultural Yield Content", yield_text, height=400)

    # Try to extract table and plot
    df_yield = extract_table_from_text(yield_text)
    if df_yield is not None:
        numeric_cols = df_yield.select_dtypes(include=np.number)
        if not numeric_cols.empty:
            st.subheader("Agricultural Yield Graphs")
            st.bar_chart(numeric_cols)
        else:
            st.info("No numeric data found in Agricultural Yield document for graph.")
    else:
        st.info("Could not detect tabular data in Agricultural Yield document.")
