"""
Document processing and text extraction module for INVESTOSCORE.
"""

import pdfplumber
import pytesseract
import pandas as pd
from typing import Dict, List, Union
import logging

class FileLoader:
    """Handles loading and processing of different document formats."""
    
    def __init__(self, config: Dict = None):
        self.supported_formats = ['.pdf', '.xlsx', '.txt']
        self.config = config
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the file loader."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def load_document(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """
        Load and process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict containing extracted text and metadata
        """
        try:
            if file_path.endswith('.pdf'):
                return self._process_pdf(file_path)
            elif file_path.endswith('.xlsx'):
                return self._process_excel(file_path)
            elif file_path.endswith('.txt'):
                return self._process_text(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def _process_pdf(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """Process PDF files with OCR fallback."""
        text_content = []
        
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if not text and self.config.get('ocr_fallback', True):
                    # Fallback to OCR if text extraction fails
                    text = self._ocr_process(page.to_image())
                text_content.append(text)
        
        return {
            'content': '\n'.join(text_content),
            'pages': text_content,
            'format': 'pdf'
        }
    
    def _process_excel(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """Process Excel files."""
        df = pd.read_excel(file_path)
        text_content = df.to_string()
        
        return {
            'content': text_content,
            'dataframe': df,
            'format': 'xlsx'
        }
    
    def _process_text(self, file_path: str) -> Dict[str, Union[str, List[str]]]:
        """Process plain text files."""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        return {
            'content': content,
            'format': 'txt'
        }
    
    def _ocr_process(self, image) -> str:
        """Process image using OCR."""
        try:
            return pytesseract.image_to_string(image)
        except Exception as e:
            self.logger.warning(f"OCR processing failed: {str(e)}")
            return ""
