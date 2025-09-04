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
        content_lower = content.lower()
        
        for category, info in self.categories.items():
            category_score = 0
            pattern = self.keyword_patterns[category]
            
            # Start with a very low base score
            base_score = 15
            
            # Count total keyword occurrences for this category
            total_matches = len(pattern.findall(content_lower))
            if total_matches == 0:
                scores[category] = base_score * 0.3  # Severely penalize categories with no matches
                continue
            
            # Track different types of matches
            meaningful_matches = 0
            strong_patterns = 0
            weak_patterns = 0
            negative_matches = 0
            
            # Analyze each sentence for context
            for sentence in sentences:
                matches = pattern.findall(sentence.lower())
                if matches:
                    for keyword in matches:
                        context_score = self._analyze_keyword_context(keyword, sentence)
                        proximity_score = self._analyze_keyword_proximity(sentence, info['keywords'])
                        pattern_score = self._check_phrase_patterns(sentence, category)
                        
                        # Classify the match based on scores
                        if context_score > 1.2 and pattern_score > 1.2:
                            strong_patterns += 1
                        elif context_score < 0.8 or pattern_score < 0.8:
                            negative_matches += 1
                        elif context_score > 1.0:
                            meaningful_matches += 1
                        else:
                            weak_patterns += 1
                        
                        # More aggressive diminishing returns
                        impact = (
                            context_score * 0.4 +
                            proximity_score * 0.2 +
                            pattern_score * 0.4
                        ) / (1 + (category_score * 0.2))
                        
                        category_score += impact
            
            # Calculate quality metrics
            keyword_coverage = meaningful_matches / len(info['keywords'])
            quality_ratio = (strong_patterns + meaningful_matches) / (weak_patterns + negative_matches + 1)
            
            # Calculate component scores
            quality_score = min(quality_ratio * 20, 40)  # Max 40 points from quality
            coverage_score = keyword_coverage * 30        # Max 30 points from coverage
            pattern_score = min(strong_patterns * 3, 15)  # Max 15 points from strong patterns
            
            # Apply penalties
            penalty = negative_matches * 5
            
            # Combine scores with base
            final_score = max(0, base_score +
                quality_score +
                coverage_score +
                pattern_score -
                penalty
            )
            
            # Apply exponential difficulty for high scores
            if final_score > 60:
                excess = final_score - 60
                final_score = 60 + (excess * 0.5)  # Half the rate of increase above 60
            if final_score > 80:
                excess = final_score - 80
                final_score = 80 + (excess * 0.25)  # Quarter the rate of increase above 80
            
            # Apply category weight
            weighted_score = final_score * info['weight']
            scores[category] = min(weighted_score, 100)
        
        # Force differentiation between categories
        score_list = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        for i, (category, score) in enumerate(score_list):
            # Apply progressive penalties to lower-ranked categories
            if i > 0:
                penalty_factor = 1.0 - (i * 0.15)  # Each rank reduces score by 15%
                scores[category] = score * penalty_factor
        
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
        # Define positive and negative context words with weights
        positive_words = {
            'significant': 0.4, 'exceptional': 0.5, 'substantial': 0.4,
            'remarkable': 0.4, 'outstanding': 0.5, 'excellent': 0.5,
            'superior': 0.4, 'leading': 0.3, 'dominant': 0.4,
            'robust': 0.3, 'strong': 0.3, 'successful': 0.3,
            'innovative': 0.3, 'efficient': 0.2, 'improved': 0.2
        }
        negative_words = {
            'significant': -0.5, 'severe': -0.5, 'substantial': -0.5,
            'critical': -0.4, 'concerning': -0.4, 'weak': -0.4,
            'poor': -0.4, 'declining': -0.4, 'unstable': -0.4,
            'volatile': -0.3, 'challenging': -0.3, 'difficult': -0.3,
            'uncertain': -0.3, 'risky': -0.3, 'problematic': -0.3
        }
        
        # Get words around the keyword (window of 6 words before and after)
        words = sentence.lower().split()
        try:
            keyword_index = words.index(keyword.lower())
            start = max(0, keyword_index - 6)
            end = min(len(words), keyword_index + 7)
            context_words = words[start:end]
            
            # Calculate weighted sentiment score
            sentiment_score = 0
            word_count = 0
            
            for word in context_words:
                if word in positive_words:
                    sentiment_score += positive_words[word]
                    word_count += 1
                elif word in negative_words:
                    sentiment_score += negative_words[word]
                    word_count += 1
            
            if word_count == 0:
                return 0.9  # Slightly negative for neutral context
            
            # Calculate final score with stronger bias towards negative
            base_score = 0.9  # Start slightly below neutral
            sentiment_impact = sentiment_score / (word_count * 0.8)  # Increase impact
            
            # More aggressive scoring range (0.3 to 1.4)
            return max(0.3, min(1.4, base_score + sentiment_impact))
            
        except ValueError:
            return 0.9  # Slightly negative for no context
    
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
        # Define phrase patterns for each category with scores
        category_patterns = {
            'financial_health': [
                (r'strong (financial|fiscal) (performance|results)', 1.5),
                (r'(healthy|robust) (balance sheet|cash flow)', 1.4),
                (r'(significant|substantial) (revenue|profit) growth', 1.3),
                (r'declining (revenue|profit|margin)', 0.7),
                (r'weak (financial|fiscal) (performance|results)', 0.6),
                (r'(concerning|poor) (cash flow|liquidity)', 0.5)
            ],
            'valuation': [
                (r'significantly (undervalued|below) (peer|market|intrinsic) value', 1.5),
                (r'attractive (valuation|multiple) compared to peers', 1.4),
                (r'premium valuation', 0.7),
                (r'overvalued (compared|relative) to peers', 0.6),
                (r'expensive (valuation|multiple)', 0.5)
            ],
            'business_model': [
                (r'industry(-|\s)leading (technology|platform|solution)', 1.5),
                (r'unique (competitive advantage|market position)', 1.4),
                (r'proven (revenue|business) model', 1.3),
                (r'unproven business model', 0.7),
                (r'significant competition', 0.6),
                (r'losing market share', 0.5)
            ],
            'management_quality': [
                (r'exceptional (leadership|management) track record', 1.5),
                (r'proven ability to execute', 1.4),
                (r'strong governance (framework|practices)', 1.3),
                (r'management turnover', 0.7),
                (r'governance concerns', 0.6),
                (r'inexperienced (management|leadership)', 0.5)
            ],
            'market_opportunity': [
                (r'dominant market (position|share|leader)', 1.5),
                (r'rapidly expanding market opportunity', 1.4),
                (r'strong growth trajectory', 1.3),
                (r'market saturation', 0.7),
                (r'declining market share', 0.6),
                (r'intense competition', 0.5)
            ]
        }
        
        # Check for matches in relevant patterns
        patterns = category_patterns.get(category, [])
        total_score = 1.0
        matches_found = False
        
        for pattern, score in patterns:
            if re.search(pattern, sentence.lower()):
                total_score *= score
                matches_found = True
        
        # If no matches found, return neutral score
        if not matches_found:
            return 1.0
            
        # Cap the final score
        return min(max(total_score, 0.5), 1.5)
    
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
