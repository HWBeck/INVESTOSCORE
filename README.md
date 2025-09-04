# INVESTOSCORE 📈

An AI-powered investment analysis tool that provides comprehensive investment analysis using free NLP models and open-source tools.

## 🎯 Features

- **Multi-Format Document Processing**: PDF, Excel, and Text files
- **AI-Powered Analysis**: 10 investment categories with NLP
- **Sophisticated Scoring**: 1-100 scoring with confidence levels
- **Risk Assessment**: Automated risk factor identification
- **Professional Reports**: Interactive Streamlit dashboard
- **Sentiment Analysis**: Both general and financial sentiment

## � Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/HWBeck/INVESTOSCORE.git
cd INVESTOSCORE
```

2. **Set Up Environment**
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Run the App**
```bash
streamlit run app.py
```

4. **Open in Browser**
- Navigate to `http://localhost:8501`
- Upload any investment document (PDF, Excel, or Text)
- Get instant AI-powered analysis

## 📊 Analysis Categories

| Category | Weight | Description |
|----------|---------|-------------|
| Financial Health | 20% | Revenue, profit, cash flow analysis |
| Valuation | 15% | Market metrics and ratios |
| Business Model | 15% | Core operations and strategy |
| Management Quality | 10% | Leadership assessment |
| Market Opportunity | 10% | Growth potential analysis |
| Risk Profile | 10% | Risk factor evaluation |
| Competitive Position | 8% | Market standing |
| Growth Strategy | 7% | Expansion plans |
| Regulatory Compliance | 3% | Legal framework |
| ESG Factors | 2% | Environmental, social, governance |

## 🤖 Technology Stack

- **NLP Models**:
  - Sentiment: `cardiffnlp/twitter-roberta-base-sentiment-latest`
  - Financial: `ProsusAI/finbert`
  - Text Processing: NLTK, scikit-learn

- **Document Processing**:
  - PDF: pdfplumber with OCR fallback
  - Excel: pandas with openpyxl
  - Text: Advanced NLP processing

- **Frontend**:
  - Streamlit dashboard
  - Plotly visualizations
  - Interactive analysis

## 💡 Usage Examples

1. **Company Analysis**:
   - Upload annual reports
   - Get instant investment scores
   - View risk assessments

2. **Comparative Analysis**:
   - Process multiple documents
   - Compare scores across companies
   - Track changes over time

3. **Quick Assessment**:
   - Upload brief documents
   - Get rapid insights
   - Focus on key metrics

## 🛠️ Development

```bash
# Run tests
python -m unittest discover tests

# Check code style
flake8 investoscore

# Run specific test file
python -m unittest tests/test_scoring_engine.py
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📮 Contact

- **GitHub Issues**: For bug reports and feature requests
- **Discussions**: For questions and community support
