# app.py

import pandas as pd
import re
from fuzzywuzzy import fuzz
import streamlit as st

# Utility to normalize text
def normalize_text(text):
    if pd.isna(text):
        return ""
    text = str(text).upper()
    text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Highlight differences in red
def highlight_diff(input_text, match_text):
    input_words = set(input_text.split())
    match_words = match_text.split()
    result = []
    for word in match_words:
        if word not in input_words:
            result.append(f"<span style='color:red'>{word}</span>")
        else:
            result.append(word)
    return ' '.join(result)

# Streamlit UI
st.title("🔎 Historical Corrective Action Estimator")

uploaded_file = st.file_uploader("Upload Historical Excel File", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    if 'Discrepancy' not in df.columns or 'Corrective Action' not in df.columns or 'Total Hours' not in df.columns:
        st.error("Excel file must have columns: Discrepancy, Corrective Action, and Total Hours")
    else:
        df['Discrepancy_Norm'] = df['Discrepancy'].apply(normalize_text)
        df['Corrective_Norm'] = df['Corrective Action'].apply(normalize_text)

        user_discrepancy = st.text_input("Enter Discrepancy")
        user_corrective = st.text_input("Enter Corrective Action")

        if st.button("Search"):
            if not user_discrepancy or not user_corrective:
                st.warning("Please enter both fields.")
            else:
                user_discrepancy_norm = normalize_text(user_discrepancy)
                user_corrective_norm = normalize_text(user_corrective)

                # Exact Match
                exact = df[
                    (df['Discrepancy_Norm'] == user_discrepancy_norm) &
                    (df['Corrective_Norm'] == user_corrective_norm)
                ]

                st.subheader("📌 Exact Match")
                if not exact.empty:
                    st.write(exact[['Discrepancy', 'Corrective Action', 'Total Hours']])
                    avg_hours = exact['Total Hours'].mean()
                    fair_quote = int(avg_hours * 0.99)
                    st.success(f"✅ Historical Avg Hours: {avg_hours:.2f} | Fair Quote: {fair_quote} hours")
                else:
                    st.info("❌ No exact match found.")

                # Approximate Matches
                st.subheader("🔍 Approximate Matches")
                threshold = 90
                results = []

                for _, row in df.iterrows():
                    score_d = fuzz.token_set_ratio(user_discrepancy_norm, row['Discrepancy_Norm'])
                    score_c = fuzz.token_set_ratio(user_corrective_norm, row['Corrective_Norm'])
                    if score_d >= threshold and score_c >= threshold:
                        results.append({
                            "Discrepancy": highlight_diff(user_discrepancy_norm, row['Discrepancy_Norm']),
                            "Corrective Action": highlight_diff(user_corrective_norm, row['Corrective_Norm']),
                            "Total Hours": row['Total Hours']
                        })

                if results:
                    for r in results:
                        st.markdown(f"""
                        <div style='margin-bottom:10px;'>
                        <b>Discrepancy:</b> {r['Discrepancy']}<br>
                        <b>Corrective Action:</b> {r['Corrective Action']}<br>
                        <b>Total Hours:</b> {r['Total Hours']}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No close matches found.")
