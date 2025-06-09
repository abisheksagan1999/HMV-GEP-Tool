import streamlit as st
from PIL import Image
import pandas as pd
import re
from fuzzywuzzy import fuzz
from io import BytesIO

# Streamlit page config
st.set_page_config(page_title="HMV Fair Quote Tool", layout="wide")

# Load logos from root directory
col1, col2, col3 = st.columns([1, 6, 1])
with col1:
    st.image("logo1.png", width=120)
with col3:
    st.image("logo2.png", width=120)

# Page title
st.markdown("""
    <h2 style='text-align:center;'>HMV Fair Quote Validation Tool</h2>
    <hr>
""", unsafe_allow_html=True)

# Upload Excel file
uploaded_file = st.file_uploader("Upload HMV Excel File (hmv_data.xlsx format):", type=["xlsx"])
if uploaded_file:
    df = pd.read_excel(uploaded_file)

    # Normalize Corrective Action text
    def normalize_text(text):
        if pd.isna(text):
            return ""
        text = str(text).upper()
        text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', text)  # Remove dates
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['Normalized Corrective Action'] = df['Corrective Action'].apply(normalize_text)

    # Create clusters using fuzzy matching
    SIMILARITY_THRESHOLD = 90
    clusters = {}
    unique_actions = df['Normalized Corrective Action'].unique()
    for action in unique_actions:
        if not action:
            continue
        found = False
        for rep in clusters:
            if fuzz.token_set_ratio(action, rep) >= SIMILARITY_THRESHOLD:
                clusters[rep].append(action)
                found = True
                break
        if not found:
            clusters[action] = [action]

    action_to_rep = {}
    for rep, members in clusters.items():
        for m in members:
            action_to_rep[m] = rep

    df['Action Cluster'] = df['Normalized Corrective Action'].map(action_to_rep)

    # Calculate reference hours per cluster
    cluster_hours = df.groupby('Action Cluster')['Total Hours'].agg(['min', 'count']).reset_index()
    cluster_hours.rename(columns={'min': 'Reference Hours', 'count': 'Occurrences'}, inplace=True)
    df = df.merge(cluster_hours, on='Action Cluster', how='left')

    # Fair quote calculation (99% of reference, rounded down)
    df['Fair Quote (hrs)'] = (df['Reference Hours'] * 0.99).apply(lambda x: int(x))

    # Filters
    st.sidebar.header("Filters")
    all_years = sorted(df['Year'].dropna().unique())
    year_filter = st.sidebar.multiselect("Select Year(s):", options=all_years, default=all_years)
    card_filter = st.sidebar.text_input("Filter by Card Number (partial match):")
    hour_min, hour_max = st.sidebar.slider("Hour Range:", 0, int(df['Total Hours'].max()),  (0, int(df['Total Hours'].max())))

    filtered_df = df.copy()
    if year_filter:
        filtered_df = filtered_df[filtered_df['Year'].isin(year_filter)]
    if card_filter:
        filtered_df = filtered_df[filtered_df['Orig. Card #'].astype(str).str.contains(card_filter, case=False)]
    filtered_df = filtered_df[(filtered_df['Total Hours'] >= hour_min) & (filtered_df['Total Hours'] <= hour_max)]

    # Text input from user for lookup
    st.write("### Enter a Discrepancy and Corrective Action to Lookup")
    user_input = st.text_area("Corrective Action (exact or partial)")
    if user_input:
        norm_input = normalize_text(user_input)
        exact_match_df = df[df['Normalized Corrective Action'] == norm_input]
        approx_match_df = df[df['Normalized Corrective Action'].apply(lambda x: fuzz.token_set_ratio(x, norm_input) >= 85)]

        if not exact_match_df.empty:
            st.success("### Exact Matches Found")
            st.dataframe(exact_match_df[['Corrective Action', 'Total Hours', 'Reference Hours', 'Fair Quote (hrs)', 'Occurrences']])
        else:
            st.warning("No exact match found.")

        if not approx_match_df.empty:
            st.info("### Approximate Matches")
            st.dataframe(approx_match_df[['Corrective Action', 'Total Hours', 'Reference Hours', 'Fair Quote (hrs)', 'Occurrences']])

    # Download option using BytesIO to avoid crash
    to_download = filtered_df[['Orig. Card #', 'Description', 'Corrective Action', 'Total Hours', 'Reference Hours', 'Fair Quote (hrs)', 'Occurrences']]
    output = BytesIO()
    to_download.to_excel(output, index=False, engine='openpyxl')
    output.seek(0)
    st.download_button("Download Filtered Results", data=output, file_name="Fair_Quote_Results.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

else:
    st.info("Please upload the 'hmv_data.xlsx' file to begin.")
