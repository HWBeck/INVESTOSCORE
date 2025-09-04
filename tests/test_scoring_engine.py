"""
Test suite for the sophisticated scoring system.
"""

import unittest
from investoscore.scoring_engine import ScoringEngine, ScoreResult
from investoscore.content_analyzer import ContentAnalysis
from typing import Dict, List

class MockContentAnalysis:
    """Mock content analysis for testing."""
    
    def __init__(self,
                 categories: Dict[str, float] = None,
                 sentiment: Dict[str, float] = None,
                 keywords: Dict[str, List[str]] = None,
                 confidence: float = 0.8,
                 entities: List[Dict] = None):
        """Initialize with test data."""
        self.categories = categories or {}
        self.sentiment = sentiment or {'overall': 0.5, 'financial': 0.5}
        self.keywords = keywords or {}
        self.confidence = confidence
        self.entities = entities or []

class TestScoringEngine(unittest.TestCase):
    """Test cases for ScoringEngine class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.engine = ScoringEngine()
        
        # Create positive test data
        cls.positive_analysis = MockContentAnalysis(
            categories={
                'financial_health': 85,
                'valuation': 75,
                'business_model': 80,
                'management_quality': 70,
                'market_opportunity': 85,
                'risk_profile': 70,
                'competitive_position': 75,
                'growth_strategy': 80,
                'regulatory_compliance': 90,
                'esg_factors': 85
            },
            sentiment={
                'overall': 0.8,
                'financial': 0.75
            },
            keywords={
                'financial_health': ['revenue', 'profit', 'margin'],
                'valuation': ['p/e ratio', 'market cap'],
                'business_model': ['strategy', 'operations'],
                'risk_profile': ['competition']
            },
            confidence=0.9,
            entities=[
                {'type': 'monetary', 'value': '$100M'},
                {'type': 'percentage', 'value': '15%'}
            ]
        )
        
        # Create negative test data
        cls.negative_analysis = MockContentAnalysis(
            categories={
                'financial_health': 35,
                'valuation': 40,
                'business_model': 30,
                'management_quality': 45,
                'market_opportunity': 30,
                'risk_profile': 25,
                'competitive_position': 35,
                'growth_strategy': 30,
                'regulatory_compliance': 40,
                'esg_factors': 35
            },
            sentiment={
                'overall': 0.3,
                'financial': 0.25
            },
            keywords={
                'risk_profile': ['litigation', 'threat', 'competition'],
                'financial_health': ['losses', 'debt']
            },
            confidence=0.6
        )
    
    def test_positive_case(self):
        """Test scoring with positive data."""
        result = self.engine.generate_score(self.positive_analysis)
        
        self.assertIsInstance(result, ScoreResult)
        self.assertTrue(65 <= result.final_score <= 100)
        self.assertIn('Strong Buy', result.recommendations[0])
        self.assertTrue(0.7 <= result.data_completeness <= 1.0)
        
        # Check category scores
        for category, score in result.category_scores.items():
            self.assertTrue(0 <= score <= 100)
    
    def test_negative_case(self):
        """Test scoring with negative data."""
        result = self.engine.generate_score(self.negative_analysis)
        
        self.assertIsInstance(result, ScoreResult)
        self.assertTrue(0 <= result.final_score <= 45)
        self.assertIn('Sell', result.recommendations[0])
        self.assertTrue(len(result.risk_factors) >= 2)
    
    def test_confidence_levels(self):
        """Test confidence level calculation."""
        result = self.engine.generate_score(self.positive_analysis)
        
        for category, level in result.confidence_levels.items():
            self.assertIn(level, ['high', 'medium', 'low'])
    
    def test_score_breakdown(self):
        """Test score breakdown details."""
        result = self.engine.generate_score(self.positive_analysis)
        
        self.assertIn('categories', result.score_breakdown)
        self.assertIn('risk_assessment', result.score_breakdown)
        self.assertIn('scoring_factors', result.score_breakdown)
        
        # Check category breakdown
        for category in result.score_breakdown['categories']:
            breakdown = result.score_breakdown['categories'][category]
            self.assertIn('original_score', breakdown)
            self.assertIn('adjusted_score', breakdown)
            self.assertIn('weight', breakdown)
            self.assertIn('weighted_contribution', breakdown)
    
    def test_empty_analysis(self):
        """Test handling of empty analysis."""
        empty_analysis = MockContentAnalysis(
            categories={},
            keywords={},
            confidence=0.0
        )
        
        result = self.engine.generate_score(empty_analysis)
        
        self.assertEqual(result.final_score, 0.0)
        self.assertTrue(result.data_completeness < 0.3)
        self.assertTrue(any('research' in r.lower() for r in result.recommendations))
    
    def test_risk_factors(self):
        """Test risk factor identification."""
        result = self.engine.generate_score(self.negative_analysis)
        
        self.assertTrue(len(result.risk_factors) > 0)
        self.assertIn('Negative financial sentiment detected', result.risk_factors)
    
    def test_sentiment_impact(self):
        """Test sentiment impact on scores."""
        # Create analysis with neutral vs positive sentiment
        neutral_analysis = MockContentAnalysis(
            categories={'financial_health': 70},
            sentiment={'overall': 0.5, 'financial': 0.5}
        )
        
        positive_analysis = MockContentAnalysis(
            categories={'financial_health': 70},
            sentiment={'overall': 0.9, 'financial': 0.9}
        )
        
        neutral_result = self.engine.generate_score(neutral_analysis)
        positive_result = self.engine.generate_score(positive_analysis)
        
        # Positive sentiment should result in higher scores
        self.assertTrue(
            positive_result.category_scores['financial_health'] >
            neutral_result.category_scores['financial_health']
        )

if __name__ == '__main__':
    unittest.main()
