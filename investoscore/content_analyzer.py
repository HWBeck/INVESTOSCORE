"""
AI-powered content analysis module for investment documents.
"""

from typing import Dict, List, Tuple
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)
from dataclasses import dataclass
import logging
import re
from collections import defaultdict

@dataclass
class ContentAnalysis:
    """Data class to store content analysis results."""
    categories: Dict[str, float]  # Category scores
    sentiment: Dict[str, float]   # Sentiment analysis results
    keywords: Dict[str, List[str]]  # Keywords found per category
    confidence: float             # Overall confidence score
    entities: List[Dict]          # Extracted entities
    summary: str                  # Brief summary of findings

class ContentAnalyzer:
    """
    Analyzes document content using NLP and ML techniques.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self._load_models()
        self._setup_categories()
        self._setup_keywords()
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    def _load_models(self):
        """Load required NLP models."""
        try:
            # Load sentiment analysis models
            self.logger.info("Loading sentiment analysis models...")
            self.general_sentiment = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.financial_sentiment = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
            
            self.logger.info("Models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def _setup_categories(self):
        """Set up investment categories and their weights."""
        self.categories = {
            'financial_health': {
                'weight': 0.20,
                'keywords': [
                    'revenue', 'profit', 'margin', 'cash flow', 'balance sheet',
                    'assets', 'liabilities', 'earnings', 'ebitda', 'net income'
                ]
            },
            'valuation': {
                'weight': 0.15,
                'keywords': [
                    'p/e ratio', 'market cap', 'valuation', 'multiple',
                    'fair value', 'intrinsic value', 'book value', 'equity value'
                ]
            },
            'business_model': {
                'weight': 0.15,
                'keywords': [
                    'business model', 'revenue model', 'operations', 'strategy',
                    'product', 'service', 'customer', 'market share'
                ]
            },
            'management_quality': {
                'weight': 0.10,
                'keywords': [
                    'ceo', 'executive', 'management team', 'leadership',
                    'experience', 'track record', 'board of directors'
                ]
            },
            'market_opportunity': {
                'weight': 0.10,
                'keywords': [
                    'market size', 'tam', 'growth potential', 'opportunity',
                    'market share', 'competitive advantage', 'expansion'
                ]
            },
            'risk_profile': {
                'weight': 0.10,
                'keywords': [
                    'risk', 'uncertainty', 'threat', 'competition', 'regulatory',
                    'litigation', 'volatility', 'exposure', 'dependency'
                ]
            },
            'competitive_position': {
                'weight': 0.08,
                'keywords': [
                    'competitor', 'market position', 'moat', 'advantage',
                    'differentiation', 'market leader', 'market share'
                ]
            },
            'growth_strategy': {
                'weight': 0.07,
                'keywords': [
                    'growth', 'expansion', 'acquisition', 'development',
                    'innovation', 'r&d', 'new market', 'scaling'
                ]
            },
            'regulatory_compliance': {
                'weight': 0.03,
                'keywords': [
                    'regulation', 'compliance', 'legal', 'regulatory',
                    'license', 'permit', 'certification', 'audit'
                ]
            },
            'esg_factors': {
                'weight': 0.02,
                'keywords': [
                    'environmental', 'social', 'governance', 'sustainable',
                    'esg', 'carbon', 'diversity', 'ethical', 'responsible'
                ]
            }
        }
    
    def _setup_keywords(self):
        """Set up regex patterns for keyword matching."""
        self.keyword_patterns = {}
        for category, info in self.categories.items():
            patterns = []
            for keyword in info['keywords']:
                # Create case-insensitive pattern, handle plural forms
                pattern = r'\b' + re.escape(keyword) + r'(?:s|es)?\b'
                patterns.append(pattern)
            self.keyword_patterns[category] = re.compile('|'.join(patterns), re.IGNORECASE)
    
    def analyze_content(self, content: str) -> ContentAnalysis:
        """
        Analyze document content and return comprehensive analysis.
        
        Args:
            content: Document content to analyze
            
        Returns:
            ContentAnalysis object with analysis results
        """
        try:
            # 1. Perform sentiment analysis
            sentiment = self._analyze_sentiment(content)
            
            # 2. Classify content into categories
            categories = self._classify_categories(content)
            
            # 3. Extract keywords
            keywords = self._extract_keywords(content)
            
            # 4. Extract entities (company names, numbers, dates)
            entities = self._extract_entities(content)
            
            # 5. Generate summary
            summary = self._generate_summary(categories, sentiment, keywords)
            
            # 6. Calculate confidence
            confidence = self._calculate_confidence(categories, sentiment, keywords)
            
            return ContentAnalysis(
                categories=categories,
                sentiment=sentiment,
                keywords=keywords,
                confidence=confidence,
                entities=entities,
                summary=summary
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {str(e)}")
            raise
    
    def _analyze_sentiment(self, content: str) -> Dict[str, float]:
        """Analyze sentiment using both general and financial models."""
        if not content.strip():
            return {
                'general': 0.5,
                'financial': 0.5,
                'overall': 0.5
            }
            
        # Split content into chunks of 512 tokens for model processing
        chunks = [content[i:i+512] for i in range(0, len(content), 512)]
        
        general_results = []
        financial_results = []
        
        for chunk in chunks:
            if chunk.strip():
                general_results.append(self.general_sentiment(chunk)[0])
                financial_results.append(self.financial_sentiment(chunk)[0])
        
        # Handle empty results
        if not general_results or not financial_results:
            return {
                'general': 0.5,
                'financial': 0.5,
                'overall': 0.5
            }
        
        # Aggregate results
        general_sentiment = sum(float(r['score']) for r in general_results) / len(general_results)
        financial_sentiment = sum(float(r['score']) for r in financial_results) / len(financial_results)
        
        return {
            'general': general_sentiment,
            'financial': financial_sentiment,
            'overall': (general_sentiment + financial_sentiment) / 2
        }
    
    def _classify_categories(self, content: str) -> Dict[str, float]:
        """Classify content into investment categories."""
        scores = {}
        
        for category, info in self.categories.items():
            # Count keyword matches
            matches = len(re.findall(self.keyword_patterns[category], content))
            # Normalize score based on content length and category weight
            base_score = matches / (len(content.split()) + 1) * 100
            scores[category] = min(base_score * info['weight'] * 100, 100)
        
        return scores
    
    def _extract_keywords(self, content: str) -> Dict[str, List[str]]:
        """Extract relevant keywords for each category."""
        keywords = defaultdict(list)
        
        for category, pattern in self.keyword_patterns.items():
            matches = pattern.findall(content)
            keywords[category] = list(set(matches))  # Remove duplicates
        
        return dict(keywords)
    
    def _extract_entities(self, content: str) -> List[Dict]:
        """Extract named entities from content."""
        # Simple regex-based entity extraction
        entities = []
        
        # Extract monetary values
        money_pattern = r'\$\d+(?:\.\d+)?(?:M|B|T)?'
        for match in re.finditer(money_pattern, content):
            entities.append({
                'type': 'monetary',
                'value': match.group(),
                'position': match.start()
            })
        
        # Extract percentages
        percentage_pattern = r'\d+(?:\.\d+)?%'
        for match in re.finditer(percentage_pattern, content):
            entities.append({
                'type': 'percentage',
                'value': match.group(),
                'position': match.start()
            })
        
        # Extract dates (simple pattern)
        date_pattern = r'\d{4}(?:-\d{2})?(?:-\d{2})?'
        for match in re.finditer(date_pattern, content):
            entities.append({
                'type': 'date',
                'value': match.group(),
                'position': match.start()
            })
        
        return sorted(entities, key=lambda x: x['position'])
    
    def _generate_summary(self, categories: Dict[str, float], 
                         sentiment: Dict[str, float], 
                         keywords: Dict[str, List[str]]) -> str:
        """Generate a brief summary of the analysis."""
        # Get top 3 categories
        top_categories = sorted(
            categories.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]
        
        # Determine overall sentiment
        sentiment_label = "positive" if sentiment['overall'] > 0.6 else \
                         "negative" if sentiment['overall'] < 0.4 else "neutral"
        
        summary = f"Analysis indicates {sentiment_label} sentiment ({sentiment['overall']:.2%}). "
        summary += f"Top categories: {', '.join(f'{cat} ({score:.1f}%)' for cat, score in top_categories)}. "
        
        return summary
    
    def _calculate_confidence(self, categories: Dict[str, float], 
                            sentiment: Dict[str, float], 
                            keywords: Dict[str, List[str]]) -> float:
        """Calculate overall confidence in the analysis."""
        # For empty content, return 0 confidence
        if not any(categories.values()) and not any(keywords.values()):
            return 0.0
            
        factors = [
            bool(categories),  # Has category scores
            bool(sentiment),   # Has sentiment scores
            any(keywords.values()),  # Found keywords
            sentiment['overall'] != 0.5,  # Clear sentiment
            max(categories.values()) > 50  # Strong category match
        ]
        
        return sum(factors) / len(factors)
