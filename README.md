# 🧩 Customer Segmentation AI

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://utkarshsolanki07-ai-ml-segementation-segmentation-1hmfr7.streamlit.app/)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.1%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Intelligent customer segmentation powered by K-Means clustering** — Unlock actionable insights from customer behavior data with an interactive web application.

---

## 🎯 Overview

This project implements a **production-ready customer segmentation system** using machine learning. It analyzes customer attributes (age, income, spending patterns, purchase behavior) to automatically group customers into meaningful segments for targeted marketing, personalization, and business strategy.

### Key Features

✨ **Interactive Web Interface** — Real-time predictions via Streamlit  
📊 **Batch Processing** — Segment thousands of customers from CSV files  
🔍 **Cluster Analysis** — Visualize cluster profiles and customer distances  
🚀 **Pre-trained Model** — Ready-to-use K-Means clustering model  
📈 **Feature Engineering** — Automatic derivation of missing features  
⚡ **Production-Ready** — Robust error handling and data validation  

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/customer-segmentation-ai.git
   cd customer-segmentation-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**
   ```bash
   streamlit run segmentation.py
   ```

4. **Open in browser**
   ```
   Local URL: http://localhost:8501
   ```

---

## 📋 Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.28 | Interactive web framework |
| `pandas` | ≥1.5 | Data manipulation & analysis |
| `numpy` | ≥1.23 | Numerical computing |
| `scikit-learn` | ≥1.1 | Machine learning algorithms |
| `joblib` | ≥1.2 | Model serialization |
| `matplotlib` | ≥3.5 | Data visualization |
| `seaborn` | ≥0.11 | Statistical visualization |

---

## 📁 Project Structure

```
customer-segmentation-ai/
├── segmentation.py              # Main Streamlit application
├── Analysis_Model.ipynb         # Jupyter notebook with model training & analysis
├── customer_segmentation.csv    # Sample dataset
├── kmeans_model.pkl             # Pre-trained K-Means model
├── scaler.pkl                   # Feature scaler (StandardScaler)
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## 🎮 How to Use

### 1️⃣ Single Customer Prediction

Predict the segment for an individual customer:

1. Navigate to the **"Single Prediction"** tab
2. Enter customer attributes:
   - Age (18-100)
   - Income (0-500,000)
   - Total Spending (0-100,000)
   - Number of Web Purchases
   - Number of Store Purchases
   - Web Visits per Month
   - Recency (days since last purchase)
3. Click **"Predict Segment"**
4. View results including:
   - Predicted cluster assignment
   - Distance to all cluster centers
   - Comparison with cluster centroid
   - Segment insights

### 2️⃣ Batch Segmentation (CSV Upload)

Segment multiple customers at once:

1. Navigate to the **"Batch Prediction (CSV)"** tab
2. Upload a CSV file with customer data
3. The app supports two formats:
   - **Pre-processed**: Age, Income, Total_Spending, NumWebPurchases, NumStorePurchases, NumWebVisitsMonth, Recency
   - **Raw data**: Year_Birth, Mnt* columns (app auto-derives missing features)
4. View cluster distribution and sample results
5. Download segmented data with cluster assignments

### 3️⃣ Cluster Profiles

Explore cluster characteristics:

1. Navigate to the **"Cluster Profiles"** tab
2. View cluster centers in original feature space
3. Analyze normalized profiles (0-1 scale)
4. Read segment descriptions and insights

### 4️⃣ Help & Documentation

Access usage guide and technical details in the **"Help / About"** tab.

---

## 🧠 Model Details

### Algorithm: K-Means Clustering

- **Number of Clusters**: 6 (configurable)
- **Features Used**: 7 key customer attributes
- **Preprocessing**: StandardScaler normalization
- **Distance Metric**: Euclidean distance

### Feature Engineering

The application automatically computes missing features:

| Feature | Derivation |
|---------|-----------|
| `Age` | `current_year - Year_Birth` |
| `Total_Spending` | Sum of MntWines, MntFruits, MntMeatProducts, MntFishProducts, MntSweetProducts, MntGoldProds |

### Cluster Descriptions

| Cluster | Profile |
|---------|---------|
| 0 | High-value, store-loyal, very recent |
| 1 | Low spend, lapsed browsers |
| 2 | Very low spend, price-sensitive but somewhat recent |
| 3 | Previously high-value omni-channel, now lapsed |
| 4 | Highest income, high spend, store-heavy, lapsed |
| 5 | Mid-value, digitally engaged, fairly recent |

---

## 📊 Data Format

### Input CSV Format (Pre-processed)

```csv
Age,Income,Total_Spending,NumWebPurchases,NumStorePurchases,NumWebVisitsMonth,Recency
35,50000,1000,10,10,3,30
42,75000,2500,15,20,5,15
28,35000,500,5,3,2,60
```

### Input CSV Format (Raw Data)

```csv
Year_Birth,Income,MntWines,MntFruits,MntMeatProducts,MntFishProducts,MntSweetProducts,MntGoldProds,NumWebPurchases,NumStorePurchases,NumWebVisitsMonth,Recency
1988,50000,100,50,200,75,25,50,10,10,3,30
1981,75000,300,100,500,150,75,100,15,20,5,15
1995,35000,50,25,100,30,10,20,5,3,2,60
```

### Output CSV Format

```csv
Age,Income,Total_Spending,NumWebPurchases,NumStorePurchases,NumWebVisitsMonth,Recency,Cluster
35,50000,1000,10,10,3,30,2
42,75000,2500,15,20,5,15,4
28,35000,500,5,3,2,60,1
```

---

## 🔧 Technical Architecture

### Data Pipeline

```
Raw Input Data
    ↓
Feature Engineering (Age, Total_Spending derivation)
    ↓
Feature Validation & Alignment
    ↓
StandardScaler Normalization
    ↓
K-Means Prediction
    ↓
Cluster Assignment & Insights
```

### Key Functions

| Function | Purpose |
|----------|---------|
| `load_artifacts()` | Load pre-trained scaler and K-Means model |
| `compute_age()` | Derive age from birth year |
| `compute_total_spending()` | Sum spending across product categories |
| `prepare_features()` | Engineer and align features to required format |
| `scale_features()` | Normalize features using StandardScaler |
| `predict_clusters()` | Generate cluster predictions |
| `get_cluster_descriptions()` | Retrieve segment insights |

---

## 🎨 User Interface

The application features a modern, intuitive interface with:

- **Tabbed Navigation**: Organized workflows for different use cases
- **Real-time Validation**: Immediate feedback on data quality
- **Interactive Visualizations**: Bar charts for cluster distances and distributions
- **Data Tables**: Formatted displays of cluster centers and comparisons
- **Download Functionality**: Export segmented results as CSV
- **Status Indicators**: Clear success/error messaging

---

## 📈 Use Cases

### 🎯 Marketing & Sales
- **Targeted Campaigns**: Customize messaging for each segment
- **Pricing Strategy**: Adjust pricing based on segment value
- **Product Recommendations**: Tailor offerings to segment preferences

### 💼 Business Intelligence
- **Customer Lifetime Value**: Identify high-value segments
- **Churn Risk**: Detect lapsed customers for re-engagement
- **Growth Opportunities**: Find underserved segments

### 📊 Analytics & Reporting
- **Segment Profiling**: Understand customer characteristics
- **Trend Analysis**: Track segment evolution over time
- **Performance Metrics**: Measure segment-specific KPIs

---

## 🔐 Error Handling

The application includes robust error handling for:

- ✅ Missing model artifacts (scaler.pkl, kmeans_model.pkl)
- ✅ Invalid CSV formats and missing columns
- ✅ Data type mismatches and coercion errors
- ✅ Out-of-range feature values
- ✅ Null/NaN value imputation

---

## 🚀 Deployment

### Streamlit Cloud (Recommended)

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click
4. Live at: https://utkarshsolanki07-ai-ml-segementation-segmentation-1hmfr7.streamlit.app/

### Docker

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["streamlit", "run", "segmentation.py"]
```

### Local Server

```bash
streamlit run segmentation.py --server.port 8501
```

---

## 📚 Model Training

The K-Means model was trained using the `Analysis_Model.ipynb` notebook:

1. **Data Loading**: Load customer_segmentation.csv
2. **Exploratory Analysis**: Understand feature distributions
3. **Feature Engineering**: Create derived features
4. **Scaling**: Apply StandardScaler normalization
5. **Model Training**: Fit K-Means with optimal cluster count
6. **Evaluation**: Analyze cluster quality and silhouette scores
7. **Serialization**: Save scaler and model as .pkl files

To retrain the model:

```bash
jupyter notebook Analysis_Model.ipynb
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Utkarsh Solanki**

- 🔗 GitHub: [@utkarshsolanki07](https://github.com/utkarshsolanki07)
- 📧 Email: [your-email@example.com](mailto:your-email@example.com)
- 💼 LinkedIn: [Your LinkedIn Profile](https://linkedin.com/in/yourprofile)

---

## 🎓 Learning Resources

- [K-Means Clustering Guide](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Customer Segmentation Best Practices](https://en.wikipedia.org/wiki/Market_segmentation)
- [Feature Scaling in ML](https://scikit-learn.org/stable/modules/preprocessing.html)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star!**

[🌐 Live Demo](https://utkarshsolanki07-ai-ml-segementation-segmentation-1hmfr7.streamlit.app/) • [📖 Documentation](#) • [🐛 Report Bug](https://github.com/yourusername/customer-segmentation-ai/issues) • [✨ Request Feature](https://github.com/yourusername/customer-segmentation-ai/issues)

</div>

---

*Last updated: 2024 | Built with ❤️ using Python, scikit-learn, and Streamlit*
