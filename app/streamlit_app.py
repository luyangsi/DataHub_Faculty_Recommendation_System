"""
Streamlit App for Dataset Recommendation System
Research question → Dataset recommendations → Analysis delivery template
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from recommends import load_catalog, recommend_datasets

# Page config
st.set_page_config(page_title="Dataset Recommender", layout="wide")

st.title("📊 Research Dataset Recommender")
st.markdown("**Input your research question → Get recommended datasets → Generate analysis template**")

# Load catalog
@st.cache_data
def load_data():
    catalog_path = Path(__file__).parent.parent / 'data' / 'catalog_datasets.csv'
    return load_catalog(catalog_path)

try:
    df_catalog = load_data()
    domains = sorted(df_catalog['domain'].dropna().unique().tolist())
except Exception as e:
    st.error(f"Error loading catalog: {e}")
    st.stop()

# Sidebar filters
st.sidebar.header("Filters")
domain_filter = st.sidebar.multiselect(
    "Domain Filter (optional)",
    options=domains,
    default=None
)

top_n = st.sidebar.slider("Number of Recommendations", 3, 10, 5)

# Main input
st.header("1️⃣ Input Research Question")
query = st.text_area(
    "Enter your research question:",
    placeholder="e.g., How do consumer complaints relate to economic conditions?",
    height=100
)

# Recommend button
if st.button("🔍 Get Recommendations", type="primary"):
    if not query.strip():
        st.warning("Please enter a research question.")
    else:
        with st.spinner("Searching datasets..."):
            recommendations = recommend_datasets(
                query=query,
                catalog_df=df_catalog,
                top_n=top_n,
                domain_filter=domain_filter if domain_filter else None
            )
        
        if recommendations.empty:
            st.warning("No matching datasets found. Try adjusting filters.")
        else:
            st.header("2️⃣ Recommended Datasets")
            
            # Display recommendations
            for idx, row in recommendations.iterrows():
                with st.expander(f"**{idx+1}. {row['dataset_name']}** (Score: {row['score']:.1f})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"**Provider:** {row['provider_or_platform']}")
                        st.markdown(f"**Domain:** {row['domain']}")
                        st.markdown(f"**Granularity:** {row['granularity']}")
                        st.markdown(f"**Coverage:** {row['coverage']}")
                        
                        # Recommendation reason
                        matched = row.get('matched_keywords', [])
                        if matched:
                            st.success(f"✓ Matched keywords: {', '.join(matched[:5])}")
                    
                    with col2:
                        st.markdown(f"**Access Level:** {row['access_level']}")
                        if pd.notna(row.get('access_note')):
                            st.info(f"📋 {row['access_note']}")
            
            # Download recommendations
            csv = recommendations.to_csv(index=False)
            st.download_button(
                label="📥 Download Recommendations (CSV)",
                data=csv,
                file_name="dataset_recommendations.csv",
                mime="text/csv"
            )
            
            # Generate delivery template
            st.header("3️⃣ Analysis Delivery Template (DSRS Style)")
            
            template = f"""
# Research Support Analysis Template

## Research Question
{query}

## Data Sources
{chr(10).join([f"- {row['dataset_name']} ({row['provider_or_platform']})" for _, row in recommendations.head(3).iterrows()])}

## Methods Checklist

### Data Cleaning
- [ ] Parse date/time columns to standard format
- [ ] Handle missing values (document approach)
- [ ] Standardize categorical variables
- [ ] Remove duplicates
- [ ] Validate data ranges and outliers

### Exploratory Data Analysis (EDA)
Recommended visualizations:
1. **Time series plot**: Trend over time (monthly/quarterly)
2. **Bar chart**: Distribution by category (top 10 categories)
3. **Correlation heatmap**: Relationship between key variables

### Statistical Analysis
- [ ] Descriptive statistics (mean, median, trends)
- [ ] Statistical test (t-test for before/after comparison OR chi-square for categorical)
- [ ] Regression model (OLS/Poisson): outcome ~ predictors + controls

### Deliverables
1. **One-page memo** with:
   - Research question
   - Data & methods summary
   - Key findings (2-3 bullets)
   - Limitations & next steps
   
2. **Dashboard/Figures**:
   - 2-3 publication-ready charts
   - Saved to `reports/figs/`
   
3. **Reproducible code**:
   - Jupyter notebook with all steps
   - Clear documentation of data sources

## Next Steps
- Collect data from recommended sources
- Implement cleaning pipeline
- Run EDA and generate visualizations
- Conduct statistical analysis
- Prepare final memo and presentation
"""
            
            st.code(template, language="markdown")
            
            st.download_button(
                label="📥 Download Template",
                data=template,
                file_name="analysis_template.md",
                mime="text/markdown"
            )

# Example queries
st.sidebar.markdown("---")
st.sidebar.subheader("Example Queries")
examples = [
    "Consumer complaints and economic trends",
    "Healthcare costs and insurance coverage",
    "Crime rates by geographic location"
]
for ex in examples:
    if st.sidebar.button(ex, key=ex):
        st.rerun()