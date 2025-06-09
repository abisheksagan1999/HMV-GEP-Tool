import streamlit as st
from PIL import Image
import pandas as pd
import re
from fuzzywuzzy import fuzz
from io import BytesIO
import time

# Streamlit page config
st.set_page_config(page_title="HMV Fair Quote Tool", layout="wide")

# Load logos from root directory and align them side-by-side
logo_col1, logo_col2, logo_col3 = st.columns([2, 6, 2])
with logo_col1:
    st.image("logo1.png", width=200)
with logo_col3:
    st.image("logo2.png", width=200)

# Page title
st.markdown("""
    <h2 style='text-align:center;'>HMV Fair Quote Validation Tool</h2>
    <hr>
""", unsafe_allow_html=True)

# Upload Excel file
uploaded_file = st.file_uploader("Upload HMV Excel File (hmv_data.xlsx format):", type=["xlsx"])

if uploaded_file:
    with st.spinner("🔄 Processing your file..."):
        placeholder = st.empty()
        for i in range(1, 101):
            time.sleep(0.01)
            placeholder.text(f"Loading... {i}%")
        placeholder.empty()

        df = pd.read_excel(uploaded_file)

        def normalize_text(text):
            if pd.isna(text):
                return ""
            text = str(text).upper()
            text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        df['Normalized Corrective Action'] = df['Corrective Action'].apply(normalize_text)
        df['Normalized Discrepancy'] = df['Description'].apply(normalize_text)
        df['Combined Key'] = df['Normalized Discrepancy'] + " | " + df['Normalized Corrective Action']

        SIMILARITY_THRESHOLD = 90
        clusters = {}
        unique_keys = df['Combined Key'].unique()
        for key in unique_keys:
            if not key:
                continue
            found = False
            for rep in clusters:
                if fuzz.token_set_ratio(key, rep) >= SIMILARITY_THRESHOLD:
                    clusters[rep].append(key)
                    found = True
                    break
            if not found:
                clusters[key] = [key]

        key_to_rep = {}
        for rep, members in clusters.items():
            for m in members:
                key_to_rep[m] = rep

        df['Cluster Key'] = df['Combined Key'].map(key_to_rep)
        cluster_hours = df.groupby('Cluster Key')['Total Hours'].agg(['min', 'count']).reset_index()
        cluster_hours.rename(columns={'min': 'Reference Hours', 'count': 'Occurrences'}, inplace=True)
        df = df.merge(cluster_hours, on='Cluster Key', how='left')
        df['Fair Quote (hrs)'] = (df['Reference Hours'] * 0.99).apply(lambda x: int(x))

        # Filters
        st.sidebar.header("Filters")
        all_years = sorted(df['Year'].dropna().unique())
        year_filter = st.sidebar.multiselect("Select Year(s):", options=all_years, default=all_years)
        card_filter = st.sidebar.text_input("Filter by Card Number (partial match):")
        hour_min, hour_max = st.sidebar.slider("Hour Range:", 0, int(df['Total Hours'].max()), (0, int(df['Total Hours'].max())))

        filtered_df = df.copy()
        if year_filter:
            filtered_df = filtered_df[filtered_df['Year'].isin(year_filter)]
        if card_filter:
            filtered_df = filtered_df[filtered_df['Orig. Card #'].astype(str).str.contains(card_filter, case=False)]
        filtered_df = filtered_df[(filtered_df['Total Hours'] >= hour_min) & (filtered_df['Total Hours'] <= hour_max)]

        # Text input boxes for discrepancy and corrective action
        st.write("### Enter Discrepancy and Corrective Action")
        with st.form("search_form"):
            discrepancy_input = st.text_input("Discrepancy")
            corrective_input = st.text_input("Corrective Action")
            submitted = st.form_submit_button("Search")

        if submitted and corrective_input:
            norm_corr = normalize_text(corrective_input)
            norm_disc = normalize_text(discrepancy_input)
            combined_input = norm_disc + " | " + norm_corr

            exact_match_df = df[df['Combined Key'] == combined_input]
            approx_match_df = df[df['Combined Key'].apply(lambda x: fuzz.token_set_ratio(x, combined_input) >= 85)]

            if not exact_match_df.empty:
                st.success("### Exact Match Found")
                st.dataframe(exact_match_df[['Description', 'Corrective Action', 'Total Hours', 'Reference Hours', 'Fair Quote (hrs)', 'Occurrences']], use_container_width=True)
            else:
                st.warning("No exact match found.")

            if not approx_match_df.empty:
                st.info("### Approximate Matches")

                def highlight_diff(row_text, input_text):
                    words1 = set(input_text.split())
                    words2 = set(row_text.split())
                    diff = words2 - words1
                    result = []
                    for word in row_text.split():
                        if word in diff:
                            result.append(f"<b><span style='color:red'>{word}</span></b>")
                        else:
                            result.append(word)
                    return " ".join(result)

                def format_row(row):
                    desc = highlight_diff(normalize_text(row['Description']), norm_disc)
                    corr = highlight_diff(normalize_text(row['Corrective Action']), norm_corr)
                    return pd.Series({
                        'Description': desc,
                        'Corrective Action': corr,
                        'Total Hours': row['Total Hours'],
                        'Reference Hours': row['Reference Hours'],
                        'Fair Quote (hrs)': row['Fair Quote (hrs)'],
                        'Occurrences': row['Occurrences']
                    })

                styled_df = approx_match_df.apply(format_row, axis=1)
                st.write("### Highlighted Differences")
                st.markdown(styled_df.to_html(escape=False, index=False), unsafe_allow_html=True)
else:
    st.info("Please upload the 'hmv_data.xlsx' file to begin.")
