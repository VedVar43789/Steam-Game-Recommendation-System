# Steam Game Recommendation System

A comprehensive machine learning-based recommendation system for Steam games that predicts user-game interactions using statistical features, collaborative filtering, and semantic embeddings.

## 📋 Overview

This project implements a progressive approach to building recommendation models, starting with simple statistical baselines and incrementally adding more sophisticated features (one-hot encoding and semantic embeddings) to improve prediction performance. The system predicts whether a user will like a game based on their preferences and game characteristics.

## 🎯 Project Goals

- **Predict user-game interactions**: Build models that predict whether a user will like a game
- **Compare feature engineering approaches**: Evaluate the impact of different feature types (statistical, one-hot encoding, semantic embeddings)
- **Understand model trade-offs**: Analyze the balance between model complexity, performance, and scalability
- **Handle cold-start scenarios**: Evaluate model performance for new users with limited interaction history

## 📊 Dataset

- **Games**: ~10,000 Steam games with metadata (genres, tags, prices, descriptions)
- **Reviews**: ~25,000 user reviews with ratings and text
- **Users**: ~9,500 unique users with varying interaction histories

## 🚀 Features

### Feature Engineering Approaches

1. **Statistical Features**: User and item statistics computed from training data only (prevents data leakage)
   - User averages, counts, ratios
   - Item averages, counts, ratios
   - Collaborative filtering features (Jaccard similarity)

2. **One-Hot Encoding (OHE)**: Binary features for user identification
   - Powerful for memorizing user preferences
   - Not scalable for large user bases
   - Struggles with cold-start scenarios

3. **SBERT Embeddings**: Pre-trained sentence transformers for semantic game understanding
   - 384-dimensional embeddings from game descriptions
   - Handles cold-start users better
   - More deployable and scalable

### Machine Learning Models

- **Logistic Regression**: Fast, interpretable baseline
- **Random Forest**: Ensemble method for non-linear patterns
- **Neural Networks**: Deep learning models for complex feature interactions

### Evaluation Metrics

- AUC-ROC (Area Under the ROC Curve)
- Precision and Recall
- Balanced Error Rate (BER)
- Precision@K and Recall@K
- Confusion matrices
- Ablation studies

## 📈 Key Results

### Best Model Performance
- **Model**: Logistic Regression + SBERT + OHE
- **AUC-ROC**: 0.9279
- **Precision**: 0.9517
- **Recall**: 0.8394
- **BER**: 0.1442

### Feature Impact Analysis

| Model | AUC-ROC | Improvement |
|-------|---------|-------------|
| Statistical Baseline | 0.7676 | Baseline |
| + OHE (LogReg) | 0.7764 | +0.0088 (+1.1%) |
| + SBERT (LogReg) | 0.7704 | +0.0027 (+0.4%) |
| **SBERT + OHE (LogReg)** | **0.9279** | **+0.1602 (+20.9%)** |

### Key Findings

- **OHE provides the largest lift** by memorizing user preferences, but it is not scalable and struggles with unseen users
- **SBERT brings semantic understanding**, handles cold-start users better, and is more deployable, though gains alone are smaller
- **Combining SBERT (content) with OHE (identity)** captures both generalization and memorization
- **Cold-start performance**: Statistical baseline achieves 0.8532 AUC for cold users (28.6% of test set)

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd Steam-Game-Recommendation-System
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Optional: GPU Support

For faster processing with SBERT embeddings, install PyTorch with CUDA support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📖 Usage

### Running the Notebook

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `workbook.ipynb` or `01_clean_sequential_notebook.ipynb`

3. Run cells sequentially to:
   - Load and explore the dataset
   - Engineer features (statistical, OHE, SBERT)
   - Train baseline models
   - Train enhanced models with additional features
   - Evaluate and compare model performance
   - Analyze feature importance and cold-start scenarios

### Notebook Structure

1. **Setup & Data Loading** - Environment setup and data preparation
2. **Exploratory Data Analysis (EDA)** - Understanding dataset characteristics
3. **Feature Engineering** - Creating statistical and semantic features
4. **Baseline Models** - Simple models with statistical features only
5. **Better Models - Add OHE** - Demonstrating impact of user identification
6. **Better Models - Add SBERT** - Demonstrating impact of semantic understanding
7. **Comprehensive Analysis** - Feature importance, ablation studies, cold-start analysis
8. **Final Summary & Conclusions** - Key findings and recommendations

## 📁 Project Structure

```
Steam-Game-Recommendation-System/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── workbook.ipynb                     # Main notebook (full analysis)
├── 01_clean_sequential_notebook.ipynb # Clean sequential version
├── steam_games.json                    # Games dataset
├── steam_new.json                   # Reviews dataset
└── video_url.txt                    # Additional resources
```

## 🔧 Dependencies

Core packages:
- `pandas>=2.0.0` - Data manipulation
- `numpy>=1.24.0` - Numerical computing
- `scikit-learn>=1.3.0` - Machine learning models
- `scipy>=1.10.0` - Scientific computing

Deep Learning:
- `torch>=2.0.0` - PyTorch framework
- `sentence-transformers>=2.2.0` - Semantic embeddings

Visualization:
- `matplotlib>=3.7.0` - Plotting
- `seaborn>=0.12.0` - Statistical visualization
- `wordcloud>=1.9.0` - Word cloud generation

Utilities:
- `networkx>=3.0` - Graph analysis
- `tqdm>=4.65.0` - Progress bars
- `jupyter>=1.0.0` - Notebook environment

## 🎓 Methodology

The project follows a progressive feature engineering approach:

1. **Baseline Models**: Start with simple statistical features (user/item averages, counts)
2. **Add OHE**: Incorporate one-hot encoding for direct user identification
3. **Add SBERT**: Integrate semantic embeddings for content-based understanding
4. **Comprehensive Analysis**: Evaluate feature importance, cold-start performance, and model trade-offs

### Key Techniques

- **Statistical Features**: Computed from training data only to prevent data leakage
- **One-Hot Encoding**: Binary features for user identification
- **SBERT Embeddings**: Pre-trained sentence transformers (`all-MiniLM-L6-v2`) for semantic game understanding
- **Collaborative Filtering**: Jaccard similarity for user-item interactions
- **Negative Sampling**: Balanced classification dataset (reviews are typically positive)

## 📝 Notes

- The dataset uses JSONL format (JSON Lines) for efficient loading
- GPU is optional but recommended for faster SBERT embedding generation
- Models are evaluated on held-out test sets with proper train/validation/test splits
- Feature engineering prevents data leakage by computing statistics only on training data

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 🙏 Acknowledgments

- Steam for providing the game data
- Sentence Transformers community for the SBERT models
- The open-source machine learning community
