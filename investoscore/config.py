"""
Configuration settings for the INVESTOSCORE system.
"""

# Model configurations
MODELS = {
    'sentiment': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
    'financial': 'ProsusAI/finbert'
}

# Investment categories and their weights
CATEGORIES = {
    'financial_health': 0.20,
    'valuation': 0.15,
    'business_model': 0.15,
    'management_quality': 0.10,
    'market_opportunity': 0.10,
    'risk_profile': 0.10,
    'competitive_position': 0.08,
    'growth_strategy': 0.07,
    'regulatory_compliance': 0.03,
    'esg_factors': 0.02
}

# Score thresholds for recommendations
SCORE_THRESHOLDS = {
    'strong_buy': 80,
    'buy': 65,
    'hold': 45,
    'sell': 35,
    'strong_sell': 0
}

# File processing settings
FILE_SETTINGS = {
    'supported_formats': ['.pdf', '.xlsx', '.txt'],
    'max_file_size_mb': 50,
    'ocr_fallback': True
}

# Processing settings
PROCESSING = {
    'batch_size': 32,
    'max_length': 512,
    'confidence_threshold': 0.7
}
