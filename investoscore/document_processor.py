"""
Enhanced document processing module with robust error handling and validation.
"""

import os
import logging
from typing import Dict, List, Union, Optional
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
from .file_loader import FileLoader

@dataclass
class ProcessingResult:
    """Data class to store document processing results."""
    content: str
    metadata: Dict
    confidence: float
    errors: List[str]
    warnings: List[str]
    completeness: float

class DocumentProcessor:
    """
    Advanced document processor with validation and error handling.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.logger = self._setup_logger()
        self.file_loader = FileLoader(config)
        self.supported_extensions = {
            '.pdf': self._process_pdf_document,
            '.xlsx': self._process_excel_document,
            '.xls': self._process_excel_document,
            '.txt': self._process_text_document
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Configure logging with detailed formatting."""
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

    def process_document(self, file_path: str) -> ProcessingResult:
        """
        Process a document with comprehensive error handling and validation.
        
        Args:
            file_path: Path to the document to process
            
        Returns:
            ProcessingResult containing extracted content and metadata
            
        Raises:
            ValueError: If file format is not supported
            FileNotFoundError: If file doesn't exist
        """
        errors = []
        warnings = []
        
        # Validate file existence
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Validate file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if file_size > 50:  # 50MB limit
            warnings.append(f"Large file detected ({file_size:.1f}MB). Processing may take longer.")
        
        # Get file extension and validate
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        try:
            # Process the document using appropriate handler
            handler = self.supported_extensions[file_ext]
            content, metadata = handler(file_path)
            
            # Calculate completeness and confidence
            completeness = self._calculate_completeness(content, metadata)
            confidence = self._calculate_confidence(content, metadata)
            
            return ProcessingResult(
                content=content,
                metadata=metadata,
                confidence=confidence,
                errors=errors,
                warnings=warnings,
                completeness=completeness
            )
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            errors.append(str(e))
            raise
    
    def _process_pdf_document(self, file_path: str) -> tuple[str, dict]:
        """Process PDF documents with OCR fallback."""
        try:
            result = self.file_loader.load_document(file_path)
            metadata = {
                'format': 'pdf',
                'ocr_used': False,
                'pages': len(result.get('pages', [])),
                'extraction_method': 'native'
            }
            
            # Check if content is extractable
            if not result['content'].strip():
                self.logger.info("No text content found, falling back to OCR")
                result = self.file_loader._ocr_process(file_path)
                metadata['ocr_used'] = True
                metadata['extraction_method'] = 'ocr'
            
            return result['content'], metadata
            
        except Exception as e:
            self.logger.error(f"PDF processing error: {str(e)}")
            raise
    
    def _process_excel_document(self, file_path: str) -> tuple[str, dict]:
        """Process Excel documents with enhanced metadata."""
        try:
            result = self.file_loader.load_document(file_path)
            df = result['dataframe']
            
            metadata = {
                'format': 'excel',
                'sheets': len(pd.ExcelFile(file_path).sheet_names),
                'rows': df.shape[0],
                'columns': df.shape[1],
                'numeric_columns': df.select_dtypes(include=['number']).columns.tolist()
            }
            
            return result['content'], metadata
            
        except Exception as e:
            self.logger.error(f"Excel processing error: {str(e)}")
            raise
    
    def _process_text_document(self, file_path: str) -> tuple[str, dict]:
        """Process text documents with metadata."""
        try:
            result = self.file_loader.load_document(file_path)
            
            # Calculate basic text statistics
            lines = result['content'].count('\n') + 1
            words = len(result['content'].split())
            
            metadata = {
                'format': 'text',
                'lines': lines,
                'words': words,
                'size_kb': os.path.getsize(file_path) / 1024
            }
            
            return result['content'], metadata
            
        except Exception as e:
            self.logger.error(f"Text processing error: {str(e)}")
            raise
    
    def _calculate_completeness(self, content: str, metadata: Dict) -> float:
        """Calculate document completeness score."""
        if not content:
            return 0.0
            
        factors = [
            bool(content.strip()),  # Has content
            len(content.split()) > 10,  # Has meaningful content
            metadata.get('format') in ['pdf', 'excel', 'text']  # Valid format
        ]
        
        return sum(factors) / len(factors)
    
    def _calculate_confidence(self, content: str, metadata: Dict) -> float:
        """Calculate confidence score for the processed content."""
        confidence = 1.0
        
        # Reduce confidence for OCR-processed documents
        if metadata.get('ocr_used'):
            confidence *= 0.8
            
        # Reduce confidence for very short documents
        if len(content.split()) < 100:
            confidence *= 0.9
            
        # Reduce confidence for incomplete data
        if metadata.get('format') == 'excel' and metadata.get('rows', 0) < 10:
            confidence *= 0.9
            
        return confidence
