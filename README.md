# 📊 Research Dataset Recommender

An intelligent system that recommends datasets for research questions and generates analysis.

**Live Demo:** http://100.73.70.44:8506

---

## 🎯 Project Overview

This tool helps researchers:
1. **Discover datasets** relevant to their research questions
2. **Get explainable recommendations** with matched keywords
3. **Generate analysis templates** following best practices
4. **Access reproducible analysis examples** using public data

Built for research support teams, data scientists, and analysts who need to quickly identify and utilize appropriate datasets for client requests.

---

## ✨ Features

### 🔍 Smart Dataset Recommendation
- **Natural language input**: Enter research questions in plain English
- **Explainable results**: See exactly why each dataset was recommended
- **Filtering options**: Filter by domain, access level, and more
- **Exportable results**: Download recommendations as CSV

### 📋 Analysis Template Generation
Automatically generates DSRS-style deliverable templates including:
- Data cleaning checklist
- EDA visualization recommendations
- Statistical analysis approaches
- One-page memo structure

### 📊 Reproducible Analysis Example
Complete Jupyter notebook demonstrating:
- Public data analysis (CFPB Consumer Complaints)
- Data cleaning and standardization
- Time series analysis and visualization
- Statistical testing (OLS regression, t-tests)
- Publication-ready figures

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/dataset-recommender.git
cd dataset-recommender

# Install dependencies
pip install -r requirements.txt

# Set up sample data (if needed)
python create_catalog.py
```

### Run the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

Then open your browser to `http://localhost:8501`

### Run the Analysis Notebook

```bash
jupyter notebook notebooks/01_public_data_analysis.ipynb
```

---

## 📁 Project Structure

```
dataset-recommender/
├── app/
│   └── streamlit_app.py          # Main web interface
├── src/
│   ├── recommend.py              # Recommendation engine
│   └── data_utils.py             # Data preprocessing utilities
├── notebooks/
│   └── public_data_analysis.ipynb  # Example analysis
├── data/
│   ├── catalog_datasets.csv      # Dataset metadata catalog
│   └── sample_data.csv           # Sample data for analysis
├── reports/
│   ├── figs/                     # Generated figures       
├── requirements.txt
└── README.md
```

---

## 💡 Usage Examples

### Example 1: Consumer Finance Research

**Input Query:**
```
How do consumer complaints relate to economic conditions?
```

**Recommendations:**
1. ✅ **CFPB Consumer Complaints** (Score: 12.5)
   - Matched: consumer, complaints, financial, economic
   - Access: Public, updated weekly
   
2. ✅ **FRED Economic Data** (Score: 9.0)
   - Matched: economic, conditions, unemployment
   - Access: Public, free API key

**Generated Template:**
- Suggested visualizations: Time series trends, complaint distribution
- Statistical approach: OLS regression with economic controls
- Deliverables: One-page memo + 2-3 charts

### Example 2: Healthcare Analysis

**Input Query:**
```
What factors influence Medicare spending patterns?
```

**Recommendations:**
1. Medicare Provider Utilization (Public)
2. CMS Medicare Claims (Restricted, requires DUA)
3. Census ACS Demographics (Public)

---

## 🔧 How It Works

### Recommendation Algorithm

The system uses an **explainable rule-based approach**:

1. **Keyword Matching** (Weight: 3.0)
   - Matches query tokens against dataset keywords field

2. **Dataset Name** (Weight: 2.0)
   - Searches dataset titles for relevant terms

3. **Domain Matching** (Weight: 1.5)
   - Considers research domain categories

4. **Description Search** (Weight: 0.5)
   - Full-text search in dataset descriptions

5. **Access Level Boost** (×1.1)
   - Prioritizes public/free datasets

**Why rule-based?**
- ✅ Fully transparent and explainable
- ✅ No training data required
- ✅ Easy to customize and debug
- ✅ Consistent, reproducible results

---

## 📊 Sample Analysis Output

The included notebook demonstrates a complete research workflow:

### Research Question
*"How do consumer financial complaints vary over time and across product categories?"*

### Key Findings
1. **Temporal Trend**: Complaints increased by ~450/month over study period (p < 0.001)
2. **Top Categories**: Credit reporting (32%), Debt collection (18%), Mortgages (15%)
3. **Seasonal Patterns**: Peak complaint volume in Q1 and Q4

### Generated Figures
![Time Series](reports/figs/fig1_complaints_timeseries.png)
![Top Issues](reports/figs/fig2_top_issues.png)

---

## 📚 Dataset Catalog

The current catalog includes **25+ datasets** across domains:

| Domain | Example Datasets | Access Level |
|--------|-----------------|--------------|
| Consumer Finance | CFPB Complaints, HMDA Mortgages | Public |
| Economics | FRED, BLS Employment | Public |
| Healthcare | Medicare Claims, CMS Utilization | Mixed |
| Finance | Compustat, CRSP (via WRDS) | Subscription |
| Demographics | Census ACS, IPUMS | Public |

---

## 🛠️ Customization

### Adding New Datasets

Edit `data/catalog_datasets.csv`:

```csv
dataset_id,dataset_name,provider_or_platform,domain,keywords,...
26,Your New Dataset,Provider Name,Domain,"keyword1,keyword2,keyword3",...
```

### Adjusting Recommendation Weights

Edit `src/recommend.py`:

```python
# Modify scoring weights in score_by_keywords()
score += 3.0  # keyword match weight
score += 2.0  # name match weight
score += 1.5  # domain match weight
```

### Custom Analysis Templates

Edit the template generation in `app/streamlit_app.py` (line ~95)

---

## 🧪 Testing

Run basic tests:

```bash
# Test recommendation engine
python -c "from src.recommend import *; print(recommend_datasets('healthcare costs', load_catalog('data/catalog_datasets.csv')))"

# Test data utilities
python -c "from src.data_utils import *; print(clean_text('  TEST  '))"
```

---

## 📈 Future Enhancements

- [ ] ML-based semantic search using embeddings
- [ ] Integration with data provider APIs (FRED, Census)
- [ ] User feedback loop to improve recommendations
- [ ] Multi-language support
- [ ] Automated data quality assessment
- [ ] Citation generation for datasets

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---


## 📸 Screenshots

### Main Interface
<img width="3407" height="1313" alt="截屏2026-01-15 下午12 28 42" src="https://github.com/user-attachments/assets/5ac7fdd0-dc72-4ede-9425-80ca23a6ec2f" />

### Recommendation Results
<img width="3346" height="1300" alt="截屏2026-01-15 下午12 30 46" src="https://github.com/user-attachments/assets/08ea47c5-78fd-4748-8914-5979152d11ea" />


---

**⭐ If you find this useful, please star the repository!**
