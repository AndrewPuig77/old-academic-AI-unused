"""
Basic tests for the Academic AI Assistant
"""
import pytest
import os
import sys

# Add the parent directory to sys.path so we can import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_environment_setup():
    """Test that the environment is properly configured"""
    assert os.path.exists('.env'), "Environment file should exist"
    
def test_imports():
    """Test that core modules can be imported"""
    try:
        from app.core.document_processor import DocumentProcessor
        from app.core.groq_analyzer import GroqAnalyzer
        from app.utils.helpers import format_analysis_results
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import core modules: {e}")

def test_groq_api_key():
    """Test that Groq API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    groq_key = os.getenv("GROQ_API_KEY")
    assert groq_key is not None, "GROQ_API_KEY should be set"
    assert groq_key != "your_groq_api_key_here", "GROQ_API_KEY should be configured with actual key"
    assert len(groq_key) > 10, "GROQ_API_KEY should be a valid key"

def test_upload_directory():
    """Test that uploads directory exists"""
    assert os.path.exists('uploads'), "Uploads directory should exist"

def test_requirements_file():
    """Test that requirements.txt exists"""
    assert os.path.exists('requirements.txt'), "Requirements file should exist"
    
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        requirements = f.read()
        assert 'streamlit' in requirements, "Streamlit should be in requirements"
        assert 'groq' in requirements, "Groq should be in requirements"

class TestDocumentProcessor:
    """Test the DocumentProcessor class"""
    
    def test_document_processor_creation(self):
        """Test that DocumentProcessor can be instantiated"""
        try:
            from app.core.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            assert processor is not None
        except Exception as e:
            pytest.fail(f"Failed to create DocumentProcessor: {e}")

class TestGroqAnalyzer:
    """Test the GroqAnalyzer class"""
    
    def test_groq_analyzer_creation(self):
        """Test that GroqAnalyzer can be instantiated with proper API key"""
        try:
            from app.core.groq_analyzer import GroqAnalyzer
            analyzer = GroqAnalyzer()
            assert analyzer is not None
            assert analyzer.api_key is not None
        except Exception as e:
            pytest.fail(f"Failed to create GroqAnalyzer: {e}")

if __name__ == "__main__":
    pytest.main([__file__])