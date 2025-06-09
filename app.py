import streamlit as st
from PIL import Image
import pandas as pd
import re
from fuzzywuzzy import fuzz
from io import BytesIO
import time
import difflib

# Streamlit page config
st.set_page_config(page_title="HMV Fair Quote Tool", layout="wide")

# Load logos and align them side-by-side
logo_col1, logo_col2, logo_col3 = st.columns([2, 6, 2])
with logo_col1:
    st.image("logo1.png", width=200)
with logo_col3:
    st.image("logo2.png", width=200)

# Title
st.markdown("""
    <h2 style='text-align:center;'>HMV Fair Quote Validation Tool</h2>
    <hr>
""", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("Upload HMV Excel File (hmv_data.xlsx format):", type=["xlsx"])

if uploaded_file:
    with st.spinner("🔄 Processing your file..."):
        progress_bar = st.progress(0)
        for i in range(1, 101):
            time.sleep(0.01)
            progress_bar.progress(i)
        progress_bar.empty()

    df = pd.read_excel(uploaded_file)

    def normalize_text(text):
        if pd.isna(text): return ""
        text = str(text).upper()
        text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['Normalized Corrective Action'] = df['Corrective Action'].apply(normalize_text)
    df['Normalized Discrepancy'] = df['Description'].apply(lambda x: normalize_text(x.replace("(FOR REFERENCE ONLY)", "")))
    df['Combined Key'] = df['Normalized Discrepancy'] + " | " + df['Normalized Corrective Action']

    clusters = {}
    for key in df['Combined Key'].unique():
        if not key: continue
        for rep in clusters:
            if fuzz.token_set_ratio(key, rep) >= 90:
                clusters[rep].append(key)
                break
        else:
            clusters[key] = [key]

    key_to_rep = {k: r for r, lst in clusters.items() for k in lst}
    df['Cluster Key'] = df['Combined Key'].map(key_to_rep)

    hours = df.groupby('Cluster Key')['Total Hours'].agg(['min', 'count']).reset_index()
    hours.columns = ['Cluster Key', 'Reference Hours', 'Occurrences']
    df = df.merge(hours, on='Cluster Key', how='left')
    df['Fair Quote (hrs)'] = (df['Reference Hours'] * 0.99).astype(int)

    st.sidebar.header("Filters")
    all_years = sorted(df['Year'].dropna().unique())
    year_filter = st.sidebar.multiselect("Select Year(s):", all_years, default=all_years)
    card_filter = st.sidebar.text_input("Card Number (partial):")
    min_hr, max_hr = st.sidebar.slider("Hour Range", 0, int(df['Total Hours'].max()), (0, int(df['Total Hours'].max())))

    filtered_df = df[df['Year'].isin(year_filter)]
    if card_filter:
        filtered_df = filtered_df[filtered_df['Orig. Card #'].astype(str).str.contains(card_filter, case=False)]
    filtered_df = filtered_df[(filtered_df['Total Hours'] >= min_hr) & (filtered_df['Total Hours'] <= max_hr)]

    st.write("### Enter Discrepancy and Corrective Action")
    with st.form("form"):
        discrepancy_input = st.text_area("Discrepancy", height=80)
        corrective_input = st.text_area("Corrective Action", height=80)
        submit = st.form_submit_button("Search")

    if submit and discrepancy_input and corrective_input:
        norm_disc = normalize_text(discrepancy_input.replace("(FOR REFERENCE ONLY)", ""))
        norm_corr = normalize_text(corrective_input)
        combined_input = norm_disc + " | " + norm_corr

        exact = df[df['Combined Key'] == combined_input]

        def semantic_overlap(a, b):
            a, b = a.split(), b.split()
            return 100 * len(set(a) & set(b)) / max(len(set(b)), 1)

        def total_similarity(row):
            d_ov = semantic_overlap(norm_disc, row['Normalized Discrepancy'])
            c_ov = semantic_overlap(norm_corr, row['Normalized Corrective Action'])
            return (d_ov + c_ov) / 2

        df['Overlap'] = df.apply(total_similarity, axis=1)

        approx = df[(df['Overlap'] >= 50) & (df['Combined Key'] != combined_input)]
        top2 = approx.sort_values(by='Overlap', ascending=False).head(2)

        closest = df[df['Overlap'] < 50].sort_values(by='Overlap', ascending=False).head(1)

        if not exact.empty:
            st.success("### ✅ Exact Match Found")
            st.dataframe(exact[['Description', 'Corrective Action', 'Total Hours', 'Reference Hours', 'Fair Quote (hrs)', 'Occurrences']].style.set_properties(**{'white-space': 'pre-wrap'}), use_container_width=True)

        if not top2.empty:
            st.info("### 🔍 Approximate Matches (Top 2)")

            def highlight_diff(text, ref):
                ref_words = set(ref.split())
                return " ".join([f"<b><span style='color:red'>{w}</span></b>" if w not in ref_words else w for w in text.split()])

            def format_row(row):
                return pd.Series({
                    'Description': highlight_diff(row['Normalized Discrepancy'], norm_disc),
                    'Corrective Action': highlight_diff(row['Normalized Corrective Action'], norm_corr),
                    'Total Hours': row['Total Hours'],
                    'Reference Hours': row['Reference Hours'],
                    'Fair Quote (hrs)': row['Fair Quote (hrs)'],
                    'Occurrences': row['Occurrences'],
                    'Overlap %': f"{row['Overlap']:.1f}%"
                })

            formatted = top2.apply(format_row, axis=1)
            st.markdown(formatted.to_frame().T.to_html(escape=False, index=False), unsafe_allow_html=True)

        if exact.empty and top2.empty and not closest.empty:
            st.warning("### 📝 No close matches found — showing nearest reference")
            row = format_row(closest.iloc[0])
            st.markdown(pd.DataFrame([row]).to_html(escape=False, index=False), unsafe_allow_html=True)

else:
    st.info("Please upload the 'hmv_data.xlsx' file to begin.")