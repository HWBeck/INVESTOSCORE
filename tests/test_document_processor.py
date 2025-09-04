"""
Test suite for document processing functionality.
"""

import os
import unittest
from pathlib import Path
from investoscore.document_processor import DocumentProcessor

class TestDocumentProcessor(unittest.TestCase):
    """Test cases for DocumentProcessor class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment and create test files."""
        cls.processor = DocumentProcessor()
        cls.test_dir = Path("test_files")
        cls.test_dir.mkdir(exist_ok=True)
        
        # Create test files
        cls._create_test_files()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def _create_test_files(cls):
        """Create sample files for testing."""
        # Create a text file
        text_content = "Test document content.\nThis is a sample file.\nIt contains multiple lines."
        cls.text_file = cls.test_dir / "sample.txt"
        cls.text_file.write_text(text_content)
        
        # Create an Excel file
        import pandas as pd
        df = pd.DataFrame({
            'Column1': [1, 2, 3],
            'Column2': ['A', 'B', 'C']
        })
        cls.excel_file = cls.test_dir / "sample.xlsx"
        df.to_excel(cls.excel_file, index=False)
    
    def test_text_processing(self):
        """Test processing of text files."""
        result = self.processor.process_document(str(self.text_file))
        
        self.assertIsNotNone(result)
        self.assertTrue(result.content)
        self.assertEqual(result.metadata['format'], 'text')
        self.assertTrue(result.completeness > 0)
        self.assertTrue(result.confidence > 0)
    
    def test_excel_processing(self):
        """Test processing of Excel files."""
        result = self.processor.process_document(str(self.excel_file))
        
        self.assertIsNotNone(result)
        self.assertTrue(result.content)
        self.assertEqual(result.metadata['format'], 'excel')
        self.assertEqual(result.metadata['rows'], 3)
        self.assertEqual(result.metadata['columns'], 2)
    
    def test_unsupported_format(self):
        """Test handling of unsupported file formats."""
        unsupported_file = self.test_dir / "test.xyz"
        unsupported_file.touch()
        
        with self.assertRaises(ValueError):
            self.processor.process_document(str(unsupported_file))
    
    def test_nonexistent_file(self):
        """Test handling of non-existent files."""
        with self.assertRaises(FileNotFoundError):
            self.processor.process_document("nonexistent.txt")

if __name__ == '__main__':
    unittest.main()
