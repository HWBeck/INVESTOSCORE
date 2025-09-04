"""
Test script to verify the installation of key dependencies.
"""

def test_imports():
    """Test importing key dependencies."""
    imports = {
        'transformers': 'Hugging Face Transformers',
        'torch': 'PyTorch',
        'pandas': 'Pandas',
        'pdfplumber': 'PDF Plumber',
        'nltk': 'NLTK',
        'sklearn': 'Scikit-learn',
        'numpy': 'NumPy',
        'pytesseract': 'PyTesseract',
        'openpyxl': 'OpenPyXL'
    }
    
    failed_imports = []
    successful_imports = []
    
    for module, name in imports.items():
        try:
            __import__(module)
            successful_imports.append(f"✅ {name} ({module})")
        except ImportError as e:
            failed_imports.append(f"❌ {name} ({module}): {str(e)}")
    
    print("\nTesting INVESTOSCORE Dependencies:\n")
    print("\n".join(successful_imports))
    
    if failed_imports:
        print("\nFailed imports:")
        print("\n".join(failed_imports))
    else:
        print("\nAll dependencies successfully imported! 🎉")

if __name__ == "__main__":
    test_imports()
