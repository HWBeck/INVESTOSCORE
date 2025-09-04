"""
Test suite for AI-powered content analysis.
"""

import unittest
from investoscore.content_analyzer import ContentAnalyzer

class TestContentAnalyzer(unittest.TestCase):
    """Test cases for ContentAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.analyzer = ContentAnalyzer()
        
        # Sample financial text for testing
        cls.sample_text = """
        Q2 2025 Financial Results - ACME Corporation
        
        Financial Overview:
        - Revenue increased 15% YoY to $125.6M
        - EBITDA margin improved to 28.5%
        - Net profit reached $45.2M
        
        Business Highlights:
        - Successfully launched new product line
        - Expanded market share in Asia to 15%
        - Completed strategic acquisition of TechCo
        
        Risk Factors:
        - Increasing market competition
        - Regulatory changes in EU market
        - Supply chain constraints
        
        Management Update:
        - New CFO appointed with 20 years experience
        - Board expanded with two independent directors
        
        ESG Initiatives:
        - Reduced carbon emissions by 12%
        - Implemented new governance framework
        - Diversity initiatives showing positive results
        
        Growth Strategy:
        - R&D investment increased to $25M
        - Planning expansion into South America
        - New partnerships under negotiation
        
        Valuation Metrics:
        - P/E ratio: 22.5x
        - Market cap: $2.8B
        - Book value per share: $45.2
        """
    
    def test_sentiment_analysis(self):
        """Test sentiment analysis functionality."""
        result = self.analyzer.analyze_content(self.sample_text)
        
        self.assertIn('general', result.sentiment)
        self.assertIn('financial', result.sentiment)
        self.assertIn('overall', result.sentiment)
        self.assertTrue(0 <= result.sentiment['overall'] <= 1)
    
    def test_category_classification(self):
        """Test category classification."""
        result = self.analyzer.analyze_content(self.sample_text)
        
        # Check all categories are present
        expected_categories = {
            'financial_health', 'valuation', 'business_model',
            'management_quality', 'market_opportunity', 'risk_profile',
            'competitive_position', 'growth_strategy',
            'regulatory_compliance', 'esg_factors'
        }
        self.assertEqual(set(result.categories.keys()), expected_categories)
        
        # Check score ranges
        for score in result.categories.values():
            self.assertTrue(0 <= score <= 100)
    
    def test_keyword_extraction(self):
        """Test keyword extraction functionality."""
        result = self.analyzer.analyze_content(self.sample_text)
        
        # Check financial health keywords
        financial_keywords = result.keywords['financial_health']
        self.assertTrue(any('revenue' in kw.lower() for kw in financial_keywords))
        self.assertTrue(any('ebitda' in kw.lower() for kw in financial_keywords))
        
        # Check ESG keywords
        esg_keywords = result.keywords['esg_factors']
        self.assertTrue(any('carbon' in kw.lower() for kw in esg_keywords))
    
    def test_entity_extraction(self):
        """Test entity extraction."""
        result = self.analyzer.analyze_content(self.sample_text)
        
        # Check monetary values
        monetary_entities = [e for e in result.entities if e['type'] == 'monetary']
        self.assertTrue(any('$125.6M' in e['value'] for e in monetary_entities))
        
        # Check percentages
        percentage_entities = [e for e in result.entities if e['type'] == 'percentage']
        self.assertTrue(any('15%' in e['value'] for e in percentage_entities))
    
    def test_confidence_scoring(self):
        """Test confidence score calculation."""
        result = self.analyzer.analyze_content(self.sample_text)
        
        self.assertTrue(0 <= result.confidence <= 1)
        # Rich content should have high confidence
        self.assertTrue(result.confidence > 0.7)
    
    def test_empty_content(self):
        """Test handling of empty content."""
        result = self.analyzer.analyze_content("")
        
        self.assertTrue(all(score == 0 for score in result.categories.values()))
        self.assertEqual(result.confidence, 0.0)
    
    def test_summary_generation(self):
        """Test summary generation."""
        result = self.analyzer.analyze_content(self.sample_text)
        
        self.assertTrue(isinstance(result.summary, str))
        self.assertTrue(len(result.summary) > 0)
        self.assertIn("sentiment", result.summary.lower())
        self.assertIn("categories", result.summary.lower())

if __name__ == '__main__':
    unittest.main()
