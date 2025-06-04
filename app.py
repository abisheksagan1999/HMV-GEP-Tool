import streamlit as st
import pandas as pd
import re
from fuzzywuzzy import fuzz
import difflib

st.set_page_config(page_title="HMV Historical Checker", layout="wide")

st.title("🛠️ HMV Historical Maintenance Validator")

uploaded_file = st.file_uploader("📤 Upload your historical Excel file (e.g., `hmv_data.xlsx`)", type=["xlsx"])

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        df.dropna(subset=["Description", "Corrective Action", "Total Hours"], inplace=True)

        desc_input = st.text_input("🔎 Enter Discrepancy / Description:")
        corr_input = st.text_input("🔧 Enter Corrective Action:")

        def normalize(text):
            text = str(text).upper()
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        def highlight_diff(base, comp):
            seq = difflib.SequenceMatcher(None, base.split(), comp.split())
            result = ""
            for tag, i1, i2, j1, j2 in seq.get_opcodes():
                if tag == 'equal':
                    result += ' ' + ' '.join(comp.split()[j1:j2])
                else:
                    result += ' <span style="color:red">' + ' '.join(comp.split()[j1:j2]) + '</span>'
            return result.strip()

        if desc_input and corr_input:
            norm_desc = normalize(desc_input)
            norm_corr = normalize(corr_input)

            st.subheader("📌 Exact Match")
            exact_matches = df[
                (df['Description'].apply(normalize) == norm_desc) &
                (df['Corrective Action'].apply(normalize) == norm_corr)
            ]

            if not exact_matches.empty:
                common_hours = exact_matches['Total Hours'].mode()[0]
                st.success(f"✅ Found {len(exact_matches)} exact match(es)")
                st.write(f"**Historical Hours:** {common_hours}")
                st.write(f"**Suggested Fair Quote (99%):** {int(common_hours * 0.99)} hours")
                st.dataframe(exact_matches[['Orig. Card #', 'Ref #', 'Year', 'Total Hours']])
            else:
                st.warning("❌ No exact match found.")

            st.subheader("🔍 Approximate Matches")
            df['Description_Score'] = df['Description'].apply(lambda x: fuzz.token_sort_ratio(norm_desc, normalize(x)))
            df['Corrective_Score'] = df['Corrective Action'].apply(lambda x: fuzz.token_sort_ratio(norm_corr, normalize(x)))
            df['Score'] = (df['Description_Score'] + df['Corrective_Score']) / 2

            approx_matches = df[df['Score'] >= 90].sort_values(by='Score', ascending=False).head(5)

            if not approx_matches.empty:
                for _, row in approx_matches.iterrows():
                    desc_highlight = highlight_diff(norm_desc, normalize(row['Description']))
                    corr_highlight = highlight_diff(norm_corr, normalize(row['Corrective Action']))
                    st.markdown(f"**Description:** {desc_highlight}", unsafe_allow_html=True)
                    st.markdown(f"**Corrective Action:** {corr_highlight}", unsafe_allow_html=True)
                    st.write(f"Hours: **{row['Total Hours']}**, Year: {row['Year']}, Card #: {row['Orig. Card #']}")
                    st.markdown("---")
            else:
                st.info("No close approximate matches found.")

    except Exception as e:
        st.error(f"⚠️ Error reading file: {e}")

else:
    st.info("⬆️ Please upload your Excel file to begin.")
