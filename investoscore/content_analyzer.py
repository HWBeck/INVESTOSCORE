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
        """Analyze sentiment with enhanced financial context awareness."""
        if not content.strip():
            return {
                'general': 0.5,
                'financial': 0.5,
                'overall': 0.5,
                'confidence': 0.0
            }
        
        # Split content into sections for more focused analysis
        sections = self._split_into_sections(content)
        
        section_weights = {
            'executive_summary': 1.5,    # Most important
            'financial_results': 1.3,    # Very important
            'business_overview': 1.0,    # Standard weight
            'market_analysis': 1.2,      # Important
            'risks': 0.8                 # Consider but don't overweight
        }
        
        section_sentiments = []
        financial_contexts = []
        
        for section, text in sections.items():
            if not text.strip():
                continue
                
            # Split into manageable chunks
            chunks = [text[i:i+512] for i in range(0, len(text), 512)]
            
            section_general = []
            section_financial = []
            
            for chunk in chunks:
                if chunk.strip():
                    # Get both sentiment analyses
                    general_result = self.general_sentiment(chunk)[0]
                    financial_result = self.financial_sentiment(chunk)[0]
                    
                    # Look for financial context indicators
                    financial_context = self._analyze_financial_context(chunk)
                    financial_contexts.append(financial_context)
                    
                    # Weight the results based on section importance
                    weight = section_weights.get(section, 1.0)
                    section_general.append(float(general_result['score']) * weight)
                    section_financial.append(float(financial_result['score']) * weight)
            
            if section_general and section_financial:
                section_sentiments.append({
                    'section': section,
                    'general': sum(section_general) / len(section_general),
                    'financial': sum(section_financial) / len(section_financial)
                })
        
        if not section_sentiments:
            return {
                'general': 0.5,
                'financial': 0.5,
                'overall': 0.5,
                'confidence': 0.0
            }
        
        # Calculate weighted averages
        general_sentiment = sum(s['general'] for s in section_sentiments) / len(section_sentiments)
        financial_sentiment = sum(s['financial'] for s in section_sentiments) / len(section_sentiments)
        
        # Calculate confidence based on financial context
        context_confidence = sum(financial_contexts) / len(financial_contexts)
        
        # Combine sentiments with financial context bias
        overall_sentiment = (general_sentiment * 0.3 + financial_sentiment * 0.7)
        
        return {
            'general': general_sentiment,
            'financial': financial_sentiment,
            'overall': overall_sentiment,
            'confidence': context_confidence
        }
    
    def _split_into_sections(self, content: str) -> Dict[str, str]:
        """Split content into logical sections based on headers."""
        sections = {
            'executive_summary': '',
            'financial_results': '',
            'business_overview': '',
            'market_analysis': '',
            'risks': ''
        }
        
        # Common section header patterns
        patterns = {
            'executive_summary': r'(?i)(executive\s+summary|overview|highlights)',
            'financial_results': r'(?i)(financial\s+(results|performance|overview)|results\s+of\s+operations)',
            'business_overview': r'(?i)(business\s+(overview|description)|company\s+profile)',
            'market_analysis': r'(?i)(market\s+(analysis|overview)|industry\s+outlook)',
            'risks': r'(?i)(risks?(\s+factors)?|uncertainties)'
        }
        
        current_section = 'executive_summary'
        lines = content.split('\n')
        
        for line in lines:
            # Check if line is a header
            for section, pattern in patterns.items():
                if re.search(pattern, line):
                    current_section = section
                    continue
            
            # Add line to current section
            sections[current_section] += line + '\n'
        
        return sections
    
    def _analyze_financial_context(self, text: str) -> float:
        """Analyze the financial context of the text."""
        financial_indicators = {
            r'\$\d+(\.\d+)?(M|B|T)?': 1.2,  # Money amounts
            r'\d+(\.\d+)?%': 1.1,           # Percentages
            r'(?i)Q[1-4]': 1.1,             # Quarters
            r'(?i)FY\d{2,4}': 1.1,          # Fiscal years
            r'(?i)(revenue|profit|margin|eps)': 1.2,  # Financial terms
            r'(?i)(market share|growth rate)': 1.1    # Business metrics
        }
        
        context_score = 1.0
        for pattern, weight in financial_indicators.items():
            if re.search(pattern, text):
                context_score *= weight
        
        return min(context_score, 2.0)  # Cap at 2.0
    
    def _classify_categories(self, content: str) -> Dict[str, float]:
        """Classify content into investment categories with advanced analysis."""
        scores = {}
        sentences = self._split_into_sentences(content)
        
        for category, info in self.categories.items():
            category_score = 0
            pattern = self.keyword_patterns[category]
            
            # Analyze each sentence for context
            for sentence in sentences:
                # Check for keyword matches in this sentence
                matches = pattern.findall(sentence.lower())
                if matches:
                    # Get context score for each match
                    for keyword in matches:
                        context_score = self._analyze_keyword_context(keyword, sentence)
                        proximity_score = self._analyze_keyword_proximity(sentence, info['keywords'])
                        pattern_score = self._check_phrase_patterns(sentence, category)
                        
                        # Combine scores with weights
                        match_score = (
                            context_score * 0.4 +    # Context is most important
                            proximity_score * 0.3 +  # Proximity next
                            pattern_score * 0.3      # Patterns also matter
                        )
                        category_score += match_score
            
            # Normalize the score to 0-100 range
            normalized_score = min(category_score * info['weight'] * 10, 100)
            scores[category] = normalized_score
        
        return scores
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences intelligently."""
        # Handle common abbreviations and special cases
        text = re.sub(r'([A-Z][.][A-Z][.](?:[A-Z][.])?)', lambda m: m.group().replace('.', '@'), text)
        text = re.sub(r'([A-Za-z][.])([A-Z])', r'\1\n\2', text)
        text = re.sub(r'[.!?]+', r'\0\n', text)
        text = text.replace('@', '.')
        
        return [s.strip() for s in text.split('\n') if s.strip()]
    
    def _analyze_keyword_context(self, keyword: str, sentence: str) -> float:
        """Analyze the context around a keyword to determine its impact."""
        # Define positive and negative context words
        positive_words = {
            'increase', 'growth', 'profit', 'strong', 'positive', 'success',
            'improved', 'growing', 'efficient', 'leading', 'innovative',
            'opportunity', 'advantage', 'successful', 'excellent', 'robust'
        }
        negative_words = {
            'decrease', 'decline', 'loss', 'weak', 'negative', 'failure',
            'poor', 'risk', 'threat', 'challenging', 'difficult', 'concern',
            'problem', 'uncertain', 'unstable', 'volatile'
        }
        
        # Get words around the keyword (window of 5 words before and after)
        words = sentence.lower().split()
        try:
            keyword_index = words.index(keyword.lower())
            start = max(0, keyword_index - 5)
            end = min(len(words), keyword_index + 6)
            context_words = set(words[start:end])
            
            # Calculate sentiment score based on surrounding words
            positive_count = len(context_words & positive_words)
            negative_count = len(context_words & negative_words)
            
            if positive_count == 0 and negative_count == 0:
                return 1.0  # Neutral context
            
            # Return a score between 0.5 and 1.5
            base_score = 1.0
            sentiment_impact = (positive_count - negative_count) * 0.25
            return max(0.5, min(1.5, base_score + sentiment_impact))
            
        except ValueError:
            return 1.0  # Keyword not found (shouldn't happen)
    
    def _analyze_keyword_proximity(self, sentence: str, keywords: List[str]) -> float:
        """Analyze how close related keywords appear together."""
        words = sentence.lower().split()
        keyword_positions = []
        
        # Find positions of all keywords in the sentence
        for i, word in enumerate(words):
            if any(keyword.lower() in word for keyword in keywords):
                keyword_positions.append(i)
        
        if len(keyword_positions) <= 1:
            return 1.0  # No proximity bonus for single keyword
        
        # Calculate average distance between keywords
        distances = []
        for i in range(len(keyword_positions) - 1):
            distance = keyword_positions[i + 1] - keyword_positions[i]
            distances.append(distance)
        
        avg_distance = sum(distances) / len(distances)
        
        # Convert distance to score (closer = better)
        # Distance of 1-2 words = 1.5, 3-5 words = 1.25, 6+ words = 1.0
        if avg_distance <= 2:
            return 1.5
        elif avg_distance <= 5:
            return 1.25
        else:
            return 1.0
    
    def _check_phrase_patterns(self, sentence: str, category: str) -> float:
        """Check for specific phrases that indicate strong relevance."""
        # Define phrase patterns for each category
        category_patterns = {
            'financial_health': [
                r'strong (financial|fiscal) (performance|results)',
                r'(healthy|robust) (balance sheet|cash flow)',
                r'(significant|substantial) (revenue|profit) growth'
            ],
            'valuation': [
                r'(attractive|fair) valuation',
                r'(trading|priced) (at|below|above) (market|peer)',
                r'(strong|compelling) (investment|value) opportunity'
            ],
            'business_model': [
                r'(sustainable|proven|successful) business model',
                r'competitive (advantage|edge|moat)',
                r'market (leader|leading|leadership)'
            ],
            'management_quality': [
                r'experienced (management|leadership) team',
                r'track record of (success|execution)',
                r'strong (governance|leadership)'
            ],
            'market_opportunity': [
                r'(growing|expanding|large) market opportunity',
                r'(strong|favorable) market position',
                r'(significant|substantial) growth potential'
            ]
        }
        
        # Check for matches in relevant patterns
        patterns = category_patterns.get(category, [])
        matches = sum(1 for pattern in patterns 
                     if re.search(pattern, sentence.lower()))
        
        # Return score based on matches (1.0 base, up to 2.0 for strong matches)
        return 1.0 + (matches * 0.5)
    
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
