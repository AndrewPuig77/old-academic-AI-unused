"""
Tests for the Streamlit application functionality
"""
import pytest
import sys
import os

# Add the parent directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_main_py_exists():
    """Test that main.py exists"""
    assert os.path.exists('main.py'), "main.py should exist"

def test_main_py_imports():
    """Test that main.py has required imports"""
    with open('main.py', 'r', encoding='utf-8') as f:
        content = f.read()
        assert 'import streamlit as st' in content, "Should import streamlit"
        assert 'from app.core.document_processor import DocumentProcessor' in content
        assert 'from app.core.groq_analyzer import GroqAnalyzer' in content

def test_streamlit_config():
    """Test that Streamlit configuration exists"""
    assert os.path.exists('.streamlit'), "Streamlit config directory should exist"

class TestAppStructure:
    """Test the application structure"""
    
    def test_app_directory_exists(self):
        """Test that app directory exists"""
        assert os.path.exists('app'), "App directory should exist"
        
    def test_core_directory_exists(self):
        """Test that core directory exists"""
        assert os.path.exists('app/core'), "Core directory should exist"
        
    def test_utils_directory_exists(self):
        """Test that utils directory exists"""
        assert os.path.exists('app/utils'), "Utils directory should exist"

class TestFlashcardGeneration:
    """Test flashcard generation functionality"""
    
    def test_groq_analyzer_has_flashcard_method(self):
        """Test that GroqAnalyzer has generate_flashcards method"""
        from app.core.groq_analyzer import GroqAnalyzer
        
        # Check if method exists
        assert hasattr(GroqAnalyzer, 'generate_flashcards'), "GroqAnalyzer should have generate_flashcards method"
        
    def test_flashcard_method_signature(self):
        """Test that generate_flashcards method has correct signature"""
        from app.core.groq_analyzer import GroqAnalyzer
        import inspect
        
        method = getattr(GroqAnalyzer, 'generate_flashcards')
        sig = inspect.signature(method)
        
        # Check parameters
        params = list(sig.parameters.keys())
        assert 'self' in params, "Method should have self parameter"
        assert 'content' in params, "Method should have content parameter"

if __name__ == "__main__":
    pytest.main([__file__])