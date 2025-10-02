# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import docx
from PyPDF2 import PdfReader
import camelot

st.title("Food Infrastructure Tracker")

# -------------------
# Helper functions
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
    # Try lattice first, then stream flavor
    tables = camelot.read_pdf(file, pages='1-end', flavor='lattice')
    if not tables or len(tables) == 0:
        tables = camelot.read_pdf(file, pages='1-end', flavor='stream')
    if tables and len(tables) > 0:
        df = tables[0].df
        # Clean numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col].str.replace(r'[^\d.]','', regex=True), errors='coerce')
        return df
    return None

def extract_table_from_docx(file):
    text = extract_text_from_docx(file)
    try:
        df = pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)
        return df
    except:
        return None

def extract_table_from_txt(file):
    text = file.read().decode("utf-8")
    try:
        df = pd.read_csv(BytesIO(text.encode()), delim_whitespace=True)
        return df
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
    # Display text content
    if market_file.name.endswith(".pdf"):
        market_text = extract_text_from_pdf(market_file)
    elif market_file.name.endswith(".docx"):
        market_text = extract_text_from_docx(market_file)
    else:
        market_text = market_file.read().decode("utf-8")
    st.subheader("Market Access Content")
    st.text_area("Content", market_text, height=400, key="market_content")

    # Extract table and show graph
    market_file.seek(0)
    df_market = extract_table(market_file)
    if df_market is not None and not df_market.empty:
        numeric_cols = df_market.select_dtypes(include='number')
        if not numeric_cols.empty:
            st.subheader("Market Access Graph")
            st.bar_chart(numeric_cols)
        else:
            st.info("No numeric data found for graph. Please check the file formatting.")
    else:
        st.info("Could not detect a table. For best results, use CSV/Excel or a PDF with clear table lines.")

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

    # Extract table and show graph
    yield_file.seek(0)
    df_yield = extract_table(yield_file)
    if df_yield is not None and not df_yield.empty:
        numeric_cols = df_yield.select_dtypes(include='number')
        if not numeric_cols.empty:
            st.subheader("Agricultural Yield Graph")
            st.bar_chart(numeric_cols)
        else:
            st.info("No numeric data found for graph. Please check the file formatting.")
    else:
        st.info("Could not detect a table. For best results, use CSV/Excel or a PDF with clear table lines.")
