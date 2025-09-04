"""
Sophisticated scoring system for investment analysis.
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import logging
from .content_analyzer import ContentAnalysis

@dataclass
class ScoreResult:
    """Data class to store scoring results."""
    final_score: float                  # 1-100 overall score
    category_scores: Dict[str, float]   # Individual category scores
    confidence_levels: Dict[str, str]   # High/Medium/Low per category
    data_completeness: float            # 0-1 completeness score
    recommendations: List[str]          # Investment recommendations
    score_breakdown: Dict[str, Dict]    # Detailed scoring breakdown
    risk_factors: List[str]             # Identified risk factors

class ScoringEngine:
    """
    Sophisticated scoring engine that combines multiple factors
    to generate investment scores and recommendations.
    """
    
    def __init__(self):
        self.logger = self._setup_logger()
        self._setup_scoring_criteria()
        
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
    
    def _setup_scoring_criteria(self):
        """Setup scoring criteria and thresholds."""
        # Score thresholds for recommendations
        self.score_thresholds = {
            'strong_buy': 80,
            'buy': 65,
            'hold': 45,
            'sell': 35
        }
        
        # Confidence level thresholds
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.0
        }
        
        # Category weights (must sum to 1)
        self.category_weights = {
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
        
        # Risk factors weight in final score
        self.risk_weight = 0.3
        
        # Sentiment impact on scores
        self.sentiment_impact = 0.15
    
    def generate_score(self, content_analysis: ContentAnalysis) -> ScoreResult:
        """
        Generate comprehensive investment score from content analysis.
        
        Args:
            content_analysis: ContentAnalysis object with analysis results
            
        Returns:
            ScoreResult object containing scores and recommendations
        """
        try:
            # 1. Calculate base category scores
            category_scores = self._calculate_category_scores(content_analysis)
            
            # 2. Apply sentiment adjustments
            adjusted_scores = self._apply_sentiment_adjustment(
                category_scores,
                content_analysis.sentiment
            )
            
            # 3. Calculate confidence levels
            confidence_levels = self._calculate_confidence_levels(
                content_analysis,
                category_scores
            )
            
            # 4. Calculate data completeness
            completeness = self._calculate_completeness(content_analysis)
            
            # 5. Identify risk factors
            risk_factors = self._identify_risk_factors(content_analysis)
            
            # 6. Calculate final score
            final_score = self._calculate_final_score(
                adjusted_scores,
                confidence_levels,
                risk_factors
            )
            
            # 7. Generate recommendations
            recommendations = self._generate_recommendations(
                final_score,
                risk_factors,
                confidence_levels
            )
            
            # 8. Create score breakdown
            score_breakdown = self._create_score_breakdown(
                category_scores,
                adjusted_scores,
                confidence_levels,
                risk_factors
            )
            
            return ScoreResult(
                final_score=final_score,
                category_scores=adjusted_scores,
                confidence_levels=confidence_levels,
                data_completeness=completeness,
                recommendations=recommendations,
                score_breakdown=score_breakdown,
                risk_factors=risk_factors
            )
            
        except Exception as e:
            self.logger.error(f"Error generating score: {str(e)}")
            raise
    
    def _calculate_category_scores(self, 
                                 content_analysis: ContentAnalysis) -> Dict[str, float]:
        """Calculate base scores for each category."""
        scores = {}
        
        for category, weight in self.category_weights.items():
            # Get raw category score
            raw_score = content_analysis.categories.get(category, 0)
            
            # Get keyword relevance
            keyword_count = len(content_analysis.keywords.get(category, []))
            keyword_factor = min(1 + (keyword_count * 0.1), 1.5)
            
            # Calculate weighted score
            scores[category] = min(raw_score * keyword_factor, 100)
        
        return scores
    
    def _apply_sentiment_adjustment(self,
                                  category_scores: Dict[str, float],
                                  sentiment: Dict[str, float]) -> Dict[str, float]:
        """Apply sentiment-based adjustments to category scores."""
        adjusted_scores = category_scores.copy()
        
        # Convert sentiment to adjustment factor (-0.15 to +0.15)
        sentiment_factor = (sentiment['overall'] - 0.5) * 2 * self.sentiment_impact
        
        for category in adjusted_scores:
            # Apply larger adjustment to sentiment-sensitive categories
            if category in ['market_opportunity', 'growth_strategy', 'risk_profile']:
                adjustment = sentiment_factor * 1.5
            else:
                adjustment = sentiment_factor
            
            # Apply adjustment and ensure score stays in 0-100 range
            adjusted_scores[category] = max(0, min(100,
                adjusted_scores[category] * (1 + adjustment)
            ))
        
        return adjusted_scores
    
    def _calculate_confidence_levels(self,
                                   content_analysis: ContentAnalysis,
                                   category_scores: Dict[str, float]) -> Dict[str, str]:
        """Calculate confidence level for each category score."""
        confidence_levels = {}
        
        for category in category_scores:
            # Factors affecting confidence:
            # 1. Content analysis confidence
            # 2. Keyword presence
            # 3. Category score strength
            factors = [
                content_analysis.confidence,
                len(content_analysis.keywords.get(category, [])) > 0,
                category_scores[category] > 50
            ]
            
            confidence = sum(1 for f in factors if f) / len(factors)
            
            # Determine confidence level
            if confidence >= self.confidence_thresholds['high']:
                confidence_levels[category] = 'high'
            elif confidence >= self.confidence_thresholds['medium']:
                confidence_levels[category] = 'medium'
            else:
                confidence_levels[category] = 'low'
        
        return confidence_levels
    
    def _calculate_completeness(self, content_analysis: ContentAnalysis) -> float:
        """Calculate data completeness score."""
        scores = []
        
        # 1. Category coverage (50% of completeness)
        category_count = sum(1 for cat, score in content_analysis.categories.items() if score > 0)
        scores.append(category_count / len(self.category_weights))
        
        # 2. Keyword presence (25% of completeness)
        keyword_categories = sum(1 for cat in self.category_weights if content_analysis.keywords.get(cat, []))
        scores.append(keyword_categories / len(self.category_weights))
        
        # 3. Entity extraction and confidence (25% of completeness)
        quality_factors = [
            bool(content_analysis.entities),           # Has extracted entities
            content_analysis.confidence > 0.5,         # Good confidence
            len(content_analysis.keywords) > 3,        # Multiple categories with keywords
            any(score > 70 for score in content_analysis.categories.values())  # Strong category scores
        ]
        scores.append(sum(quality_factors) / len(quality_factors))
        
        # Calculate weighted average
        weights = [0.5, 0.25, 0.25]  # Category, Keyword, Quality weights
        return sum(score * weight for score, weight in zip(scores, weights))
    
    def _identify_risk_factors(self, content_analysis: ContentAnalysis) -> List[str]:
        """Identify key risk factors from the analysis."""
        risk_factors = []
        
        # Check risk-related keywords
        risk_keywords = content_analysis.keywords.get('risk_profile', [])
        if risk_keywords:
            risk_factors.extend([f"Risk identified: {kw}" for kw in risk_keywords])
        
        # Check negative sentiment
        if content_analysis.sentiment['financial'] < 0.4:
            risk_factors.append("Negative financial sentiment detected")
        
        # Check confidence levels
        low_confidence_categories = [
            cat for cat, level in content_analysis.categories.items()
            if level < 30
        ]
        if low_confidence_categories:
            risk_factors.append(
                f"Low scores in categories: {', '.join(low_confidence_categories)}"
            )
        
        return risk_factors
    
    def _calculate_final_score(self,
                             category_scores: Dict[str, float],
                             confidence_levels: Dict[str, str],
                             risk_factors: List[str]) -> float:
        """Calculate final investment score."""
        # 1. Calculate weighted category score
        weighted_score = sum(
            score * self.category_weights[category]
            for category, score in category_scores.items()
        )
        
        # 2. Apply confidence adjustment
        confidence_factor = sum(
            1.0 if level == 'high' else 0.8 if level == 'medium' else 0.6
            for level in confidence_levels.values()
        ) / len(confidence_levels)
        
        # 3. Apply risk factor penalty
        risk_penalty = len(risk_factors) * self.risk_weight
        
        # Calculate final score (0-100 range)
        final_score = max(0, min(100,
            weighted_score * confidence_factor - risk_penalty
        ))
        
        return round(final_score, 1)
    
    def _generate_recommendations(self,
                                final_score: float,
                                risk_factors: List[str],
                                confidence_levels: Dict[str, str]) -> List[str]:
        """Generate investment recommendations based on analysis."""
        recommendations = []
        
        # 1. Basic recommendation based on score
        if final_score >= self.score_thresholds['strong_buy']:
            recommendations.append("Strong Buy - Excellent investment potential")
        elif final_score >= self.score_thresholds['buy']:
            recommendations.append("Buy - Good investment potential")
        elif final_score >= self.score_thresholds['hold']:
            recommendations.append("Hold - Monitor for developments")
        else:
            recommendations.append("Sell - Consider alternative investments")
        
        # 2. Add risk warnings if necessary
        if risk_factors:
            recommendations.append(
                f"Exercise caution due to {len(risk_factors)} identified risk factors"
            )
        
        # 3. Add confidence-based recommendations
        low_confidence_cats = [
            cat for cat, level in confidence_levels.items()
            if level == 'low'
        ]
        if low_confidence_cats:
            recommendations.append(
                "Additional research recommended for: " + 
                ", ".join(low_confidence_cats)
            )
        
        return recommendations
    
    def _create_score_breakdown(self,
                              original_scores: Dict[str, float],
                              adjusted_scores: Dict[str, float],
                              confidence_levels: Dict[str, str],
                              risk_factors: List[str]) -> Dict[str, Dict]:
        """Create detailed breakdown of scoring components."""
        breakdown = {
            'categories': {
                category: {
                    'original_score': original_scores[category],
                    'adjusted_score': adjusted_scores[category],
                    'weight': self.category_weights[category],
                    'weighted_contribution': 
                        adjusted_scores[category] * self.category_weights[category],
                    'confidence': confidence_levels[category]
                }
                for category in original_scores
            },
            'risk_assessment': {
                'risk_count': len(risk_factors),
                'risk_penalty': len(risk_factors) * self.risk_weight,
                'risk_factors': risk_factors
            },
            'scoring_factors': {
                'category_weights': self.category_weights,
                'risk_weight': self.risk_weight,
                'sentiment_impact': self.sentiment_impact
            }
        }
        
        return breakdown
