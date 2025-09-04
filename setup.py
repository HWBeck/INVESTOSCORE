from setuptools import setup, find_packages

setup(
    name="investoscore",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'pandas>=1.5.0',
        'pdfplumber>=0.9.0',
        'nltk>=3.8.1',
        'scikit-learn>=1.2.0',
        'numpy>=1.23.0',
        'pytesseract>=0.3.10',
        'openpyxl>=3.1.0'
    ],
    python_requires='>=3.8',
)
