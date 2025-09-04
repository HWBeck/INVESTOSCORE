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

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Header
    st.title("📈 INVESTOSCORE")
    st.subheader("AI-Powered Investment Analysis")
    
    # Sidebar
    st.sidebar.header("Upload Documents")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a file (PDF, Excel, or Text)",
        type=['pdf', 'xlsx', 'txt']
    )
    
    # Main content area
    if uploaded_file is not None:
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                file_path = tmp_file.name
            
            try:
                with st.spinner("Processing document..."):
                    # Process document
                    doc_result = st.session_state.processor.process_document(file_path)
                    
                with st.spinner("Analyzing content..."):
                    # Analyze content
                    analysis_result = st.session_state.analyzer.analyze_content(doc_result.content)
                    st.session_state.last_analysis = analysis_result
                    
                with st.spinner("Generating investment score..."):
                    # Generate score
                    score_result = st.session_state.scoring_engine.generate_score(analysis_result)
                    st.session_state.last_score = score_result
                
                # Display Results
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Overall Score
                    st.metric(
                        "Investment Score",
                        f"{score_result.final_score:.1f}/100",
                        f"Data Completeness: {score_result.data_completeness:.1%}"
                    )
                    
                    # Recommendations
                    st.subheader("📋 Recommendations")
                    for rec in score_result.recommendations:
                        st.write(f"• {rec}")
                    
                    # Risk Factors
                    if score_result.risk_factors:
                        st.subheader("⚠️ Risk Factors")
                        for risk in score_result.risk_factors:
                            st.write(f"• {risk}")
                
                with col2:
                    # Sentiment Analysis
                    st.subheader("🎯 Sentiment Analysis")
                    sentiment_df = pd.DataFrame({
                        'Type': ['Overall', 'Financial'],
                        'Score': [
                            f"{analysis_result.sentiment['overall']:.1%}",
                            f"{analysis_result.sentiment['financial']:.1%}"
                        ]
                    })
                    st.dataframe(sentiment_df)
                
                # Category Scores
                st.subheader("📊 Category Analysis")
                col3, col4 = st.columns([1, 1])
                
                with col3:
                    # Radar Chart
                    fig = plot_category_scores(score_result.category_scores)
                    st.plotly_chart(fig)
                
                with col4:
                    # Score Table
                    st.dataframe(
                        create_score_table(score_result),
                        hide_index=True
                    )
                
                # Keywords Found
                st.subheader("🔑 Key Findings")
                for category, keywords in analysis_result.keywords.items():
                    if keywords:
                        st.write(f"**{category.replace('_', ' ').title()}:**")
                        st.write(", ".join(keywords))
                
            finally:
                # Clean up temporary file
                os.unlink(file_path)
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    
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
