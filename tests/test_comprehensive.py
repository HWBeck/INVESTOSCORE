"""
Comprehensive testing script for document processing functionality.
This script tests real-world scenarios and edge cases.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from investoscore.document_processor import DocumentProcessor

def create_test_files(test_dir: Path):
    """Create a variety of test files with realistic content."""
    
    # Create test directory if it doesn't exist
    test_dir.mkdir(exist_ok=True)
    
    # 1. Create a rich text file with financial content
    financial_text = """
    Financial Analysis Report
    Company: ACME Corporation
    Date: September 4, 2025

    1. Financial Overview
    - Revenue: $50.2M (↑12% YoY)
    - EBITDA: $15.6M (↑8% YoY)
    - Net Profit Margin: 18.5%

    2. Risk Factors
    - Market volatility remains high
    - Increasing competition in core markets
    - Regulatory changes expected in Q4

    3. Growth Opportunities
    - Expansion into Asian markets
    - New product line launch in 2026
    - Strategic acquisitions under consideration
    """
    (test_dir / "financial_report.txt").write_text(financial_text)

    # 2. Create an Excel file with financial data
    financial_data = {
        'Metric': ['Revenue', 'EBITDA', 'Net Profit', 'Operating Margin'],
        '2024': [45.8, 14.2, 8.9, 0.195],
        '2025': [50.2, 15.6, 9.8, 0.185],
        'YoY Change': ['12%', '8%', '10%', '-5%']
    }
    df = pd.DataFrame(financial_data)
    df.to_excel(test_dir / "financial_metrics.xlsx", index=False)

    # 3. Create an empty file
    (test_dir / "empty.txt").touch()

    # 4. Create a large text file
    large_text = "This is a test line.\n" * 10000
    (test_dir / "large_file.txt").write_text(large_text)

    # 5. Create a file with special characters
    special_chars = """
    Special Characters Test
    €1000 revenue
    ¥500 cost
    £750 profit
    Multiple lines with μ, α, β characters
    """
    (test_dir / "special_chars.txt").write_text(special_chars)

    # 6. Create an Excel file with multiple sheets
    with pd.ExcelWriter(test_dir / "multi_sheet.xlsx") as writer:
        # Balance Sheet
        balance_sheet = pd.DataFrame({
            'Asset': ['Cash', 'Inventory', 'Receivables'],
            'Amount': [1000000, 500000, 750000]
        })
        balance_sheet.to_excel(writer, sheet_name='Balance Sheet', index=False)
        
        # Income Statement
        income_stmt = pd.DataFrame({
            'Item': ['Revenue', 'Expenses', 'Profit'],
            'Amount': [2000000, 1500000, 500000]
        })
        income_stmt.to_excel(writer, sheet_name='Income Statement', index=False)

def run_comprehensive_tests():
    """Run comprehensive tests on the document processor."""
    
    test_dir = Path("comprehensive_test_files")
    create_test_files(test_dir)
    processor = DocumentProcessor()
    
    print("\n🔍 Running Comprehensive Document Processing Tests\n")
    
    try:
        # Test 1: Process financial text file
        print("\n📝 Testing Financial Text Processing:")
        result = processor.process_document(str(test_dir / "financial_report.txt"))
        print(f"✓ Completeness: {result.completeness:.2%}")
        print(f"✓ Confidence: {result.confidence:.2%}")
        print(f"✓ Metadata: {result.metadata}")
        
        # Test 2: Process Excel file
        print("\n📊 Testing Excel Processing:")
        result = processor.process_document(str(test_dir / "financial_metrics.xlsx"))
        print(f"✓ Rows: {result.metadata['rows']}")
        print(f"✓ Columns: {result.metadata['columns']}")
        print(f"✓ Numeric columns: {result.metadata['numeric_columns']}")
        
        # Test 3: Process empty file
        print("\n📄 Testing Empty File Handling:")
        try:
            result = processor.process_document(str(test_dir / "empty.txt"))
            print(f"✓ Completeness: {result.completeness:.2%}")
        except Exception as e:
            print(f"✓ Empty file handled: {str(e)}")
        
        # Test 4: Process large file
        print("\n📚 Testing Large File Processing:")
        result = processor.process_document(str(test_dir / "large_file.txt"))
        print(f"✓ File size: {result.metadata['size_kb']:.2f}KB")
        print(f"✓ Lines processed: {result.metadata['lines']}")
        
        # Test 5: Process special characters
        print("\n🔤 Testing Special Characters:")
        result = processor.process_document(str(test_dir / "special_chars.txt"))
        print(f"✓ Content length: {len(result.content)}")
        print(f"✓ Special chars preserved: {'€' in result.content}")
        
        # Test 6: Process multi-sheet Excel
        print("\n📑 Testing Multi-sheet Excel:")
        result = processor.process_document(str(test_dir / "multi_sheet.xlsx"))
        print(f"✓ Sheets detected: {result.metadata['sheets']}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        raise
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    run_comprehensive_tests()
