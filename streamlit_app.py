"""
Streamlit web interface for INVESTOSCORE.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
import tempfile
import os
from typing import Dict, List

from investoscore.document_processor import DocumentProcessor
from investoscore.content_analyzer import ContentAnalyzer
from investoscore.scoring_engine import ScoringEngine

# Set page config
st.set_page_config(
    page_title="INVESTOSCORE - AI Investment Analysis",
    page_icon="📈",
    layout="wide"
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = ContentAnalyzer()
    if 'scoring_engine' not in st.session_state:
        st.session_state.scoring_engine = ScoringEngine()
    if 'last_analysis' not in st.session_state:
        st.session_state.last_analysis = None
    if 'last_score' not in st.session_state:
        st.session_state.last_score = None

def plot_category_scores(scores: Dict[str, float]) -> go.Figure:
    """Create a radar chart of category scores."""
    categories = list(scores.keys())
    values = list(scores.values())
    
    fig = go.Figure(data=go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        name='Category Scores'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100]
            )
        ),
        showlegend=False
    )
    
    return fig

def create_score_table(score_result) -> pd.DataFrame:
    """Create a DataFrame of category scores and confidence levels."""
    data = []
    for category, details in score_result.score_breakdown['categories'].items():
        data.append({
            'Category': category.replace('_', ' ').title(),
            'Score': f"{details['adjusted_score']:.1f}",
            'Confidence': details['confidence'].title(),
            'Weight': f"{details['weight'] * 100:.1f}%",
            'Contribution': f"{details['weighted_contribution']:.1f}"
        })
    return pd.DataFrame(data)

def process_documents(company_name: str, uploaded_files: list):
    """Process multiple documents and aggregate results."""
    all_content = ""
    all_analyses = []
    
    for uploaded_file in uploaded_files:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            try:
                # Process document
                doc_result = st.session_state.processor.process_document(file_path)
                all_content += doc_result.content + "\n\n"
                
            finally:
                # Clean up temporary file
                os.unlink(file_path)
                
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            return None
    
    try:
        # Analyze aggregated content
        analysis_result = st.session_state.analyzer.analyze_content(all_content)
        score_result = st.session_state.scoring_engine.generate_score(analysis_result)
        
        return analysis_result, score_result
    except Exception as e:
        st.error(f"Error analyzing content: {str(e)}")
        return None

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("📈 INVESTOSCORE")
    st.subheader("AI-Powered Investment Analysis")
    
    # Sidebar
    st.sidebar.header("Company Information")
    company_name = st.sidebar.text_input("Company Name", "")
    
    st.sidebar.header("Upload Documents")
    uploaded_files = st.sidebar.file_uploader(
        "Choose files (PDF, Excel, or Text)",
        type=['pdf', 'xlsx', 'txt'],
        accept_multiple_files=True
    )
    
    # Document list display
    if uploaded_files:
        st.sidebar.subheader("📄 Uploaded Documents")
        for file in uploaded_files:
            st.sidebar.text(f"• {file.name}")
    
    # Main content area
    if uploaded_files and company_name:
        with st.spinner(f"Analyzing {len(uploaded_files)} documents for {company_name}..."):
            results = process_documents(company_name, uploaded_files)
            
            if results:
                analysis_result, score_result = results
                st.session_state.last_analysis = analysis_result
                st.session_state.last_score = score_result
                
                # Company header with metadata
                st.header(f"📊 Analysis: {company_name}")
                st.caption(f"Based on {len(uploaded_files)} documents • Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
                
                # Add company analysis summary
                st.info(f"""
                **Company Overview:**
                - Documents Analyzed: {len(uploaded_files)}
                - Analysis Confidence: {analysis_result.confidence:.1%}
                - Primary Sentiment: {'Positive' if analysis_result.sentiment['overall'] > 0.6 else 'Negative' if analysis_result.sentiment['overall'] < 0.4 else 'Neutral'}
                """)
                
                # Display Results in a single page
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.metric(
                        f"Investment Score for {company_name}",
                        f"{score_result.final_score:.1f}/100",
                        f"Data Completeness: {score_result.data_completeness:.1%}"
                    )
                    
                    st.subheader("📋 Key Recommendations")
                    for rec in score_result.recommendations:
                        st.write(f"• {rec}")
                
                with col2:
                    st.subheader("🎯 Sentiment Analysis")
                    sentiment_df = pd.DataFrame({
                        'Type': ['Overall', 'Financial'],
                        'Score': [
                            f"{analysis_result.sentiment['overall']:.1%}",
                            f"{analysis_result.sentiment['financial']:.1%}"
                        ]
                    })
                    st.dataframe(sentiment_df)

                # Category Analysis
                st.subheader("📊 Category Analysis")
                col3, col4 = st.columns([1, 1])
                
                with col3:
                    fig = plot_category_scores(score_result.category_scores)
                    st.plotly_chart(fig)
                
                with col4:
                    st.dataframe(
                        create_score_table(score_result),
                        hide_index=True
                    )

                # Document Analysis
                st.subheader("📑 Document Analysis")
                for category, keywords in analysis_result.keywords.items():
                    if keywords:
                        with st.expander(f"{category.replace('_', ' ').title()} Keywords"):
                            st.write(", ".join(keywords))
                
                if analysis_result.entities:
                    with st.expander("📌 Key Entities Detected"):
                        for entity in analysis_result.entities:
                            st.write(f"• {entity['value']} ({entity['type']})") 
    
    elif uploaded_files:
        st.warning("Please enter a company name to begin analysis.")
    elif company_name:
        st.warning("Please upload at least one document to analyze.")
    else:
        # Display welcome message
        st.info(
            "👈 Upload a document (PDF, Excel, or Text) to begin the investment analysis."
        )
        
        # Display feature list
        st.subheader("🚀 Features")
        st.write("""
        - Multi-format document processing
        - AI-powered content analysis
        - Sophisticated scoring system
        - Risk factor identification
        - Investment recommendations
        - Sentiment analysis
        - Category-wise breakdown
        """)

if __name__ == "__main__":
    main()
