"""
AI Provider Factory - Unified interface for multiple AI providers
Handles switching between Groq and Gemini seamlessly
"""

import logging
from typing import Dict, Any, Union, Optional
from abc import ABC, abstractmethod

from .groq_analyzer import GroqAnalyzer
from .gemini_analyzer import GeminiAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AIAnalyzerInterface(ABC):
    """Abstract base class for AI analyzers"""
    
    @abstractmethod
    def analyze_paper(self, paper_text: str, analysis_options: Dict[str, bool]) -> Dict[str, str]:
        """Analyze academic paper/document"""
        pass
    
    @abstractmethod
    def generate_flashcards(self, content: str) -> str:
        """Generate study flashcards"""
        pass
    
    @abstractmethod
    def create_practice_questions(self, content: str, difficulty: str = "mixed") -> str:
        """Create practice questions"""
        pass
    
    @abstractmethod
    def build_study_guide(self, content: str, focus_areas: list = None) -> str:
        """Build comprehensive study guide"""
        pass

class AIProviderFactory:
    """Factory class to create and manage AI providers"""
    
    PROVIDERS = {
        "Groq": "groq",
        "Gemini": "gemini"
    }
    
    @classmethod
    def create_analyzer(cls, provider: str) -> Union[GroqAnalyzer, GeminiAnalyzer]:
        """
        Create an analyzer instance for the specified provider.
        
        Args:
            provider: AI provider name ("groq" or "gemini")
            
        Returns:
            Analyzer instance
            
        Raises:
            ValueError: If provider is not supported
            Exception: If initialization fails
        """
        provider_lower = provider.lower()
        
        try:
            if provider_lower == "groq":
                logger.info("Initializing Groq analyzer...")
                return GroqAnalyzer()
            elif provider_lower == "gemini":
                logger.info("Initializing Gemini analyzer...")
                return GeminiAnalyzer()
            else:
                raise ValueError(f"Unsupported AI provider: {provider}. Supported providers: {list(cls.PROVIDERS.keys())}")
                
        except Exception as e:
            logger.error(f"Failed to initialize {provider} analyzer: {str(e)}")
            raise Exception(f"Could not initialize {provider} analyzer: {str(e)}")
    
    @classmethod
    def get_available_providers(cls) -> Dict[str, str]:
        """Get list of available AI providers"""
        return cls.PROVIDERS.copy()
    
    @classmethod
    def validate_provider(cls, provider: str) -> bool:
        """Check if a provider is supported"""
        return provider.lower() in [p.lower() for p in cls.PROVIDERS.values()]

class UnifiedAIAnalyzer:
    """
    Unified interface that wraps multiple AI providers.
    Allows seamless switching between providers.
    """
    
    def __init__(self, default_provider: str = "groq"):
        """
        Initialize with default provider.
        
        Args:
            default_provider: Default AI provider to use
        """
        self.current_provider = default_provider.lower()
        self.analyzers = {}  # Cache analyzers to avoid re-initialization
        self.factory = AIProviderFactory()
        
        # Initialize default provider
        try:
            self.get_analyzer(self.current_provider)
            logger.info(f"UnifiedAIAnalyzer initialized with {default_provider}")
        except Exception as e:
            logger.error(f"Failed to initialize default provider {default_provider}: {e}")
            # Try fallback to other provider
            fallback = "gemini" if default_provider.lower() == "groq" else "groq"
            try:
                self.current_provider = fallback
                self.get_analyzer(fallback)
                logger.info(f"Fell back to {fallback} provider")
            except Exception as fallback_error:
                raise Exception(f"Could not initialize any AI provider. {default_provider}: {e}, {fallback}: {fallback_error}")
    
    def get_analyzer(self, provider: str = None) -> Union[GroqAnalyzer, GeminiAnalyzer]:
        """
        Get analyzer instance for specified provider.
        
        Args:
            provider: AI provider name (uses current if None)
            
        Returns:
            Analyzer instance
        """
        if provider is None:
            provider = self.current_provider
        
        provider_lower = provider.lower()
        
        # Return cached analyzer if available
        if provider_lower in self.analyzers:
            return self.analyzers[provider_lower]
        
        # Create new analyzer and cache it
        try:
            analyzer = self.factory.create_analyzer(provider_lower)
            self.analyzers[provider_lower] = analyzer
            return analyzer
        except Exception as e:
            logger.error(f"Failed to get {provider} analyzer: {e}")
            raise
    
    def switch_provider(self, new_provider: str) -> bool:
        """
        Switch to a different AI provider.
        
        Args:
            new_provider: Name of the new provider
            
        Returns:
            True if switch successful, False otherwise
        """
        try:
            new_provider_lower = new_provider.lower()
            
            if not self.factory.validate_provider(new_provider_lower):
                logger.error(f"Invalid provider: {new_provider}")
                return False
            
            # Test if we can get the analyzer (this will initialize if needed)
            analyzer = self.get_analyzer(new_provider_lower)
            
            # If successful, update current provider
            self.current_provider = new_provider_lower
            logger.info(f"Successfully switched to {new_provider} provider")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch to {new_provider}: {e}")
            return False
    
    def get_current_provider(self) -> str:
        """Get the name of the current AI provider"""
        return self.current_provider.capitalize()
    
    def get_available_providers(self) -> list:
        """Get list of available providers"""
        return list(self.factory.get_available_providers().keys())
    
    def get_provider_status(self) -> Dict[str, bool]:
        """
        Check which providers are available and working.
        
        Returns:
            Dictionary with provider names and their status
        """
        status = {}
        
        for provider_name, provider_key in self.factory.get_available_providers().items():
            try:
                # Try to get analyzer (will initialize if needed)
                self.get_analyzer(provider_key)
                status[provider_name] = True
            except Exception as e:
                logger.warning(f"{provider_name} not available: {e}")
                status[provider_name] = False
        
        return status
    
    # Unified interface methods that delegate to current provider
    
    def analyze_paper(self, paper_text: str, analysis_options: Dict[str, bool]) -> Dict[str, str]:
        """
        Analyze academic paper using current provider.
        
        Args:
            paper_text: Text content of the paper
            analysis_options: Dictionary of analysis options
            
        Returns:
            Analysis results
        """
        try:
            analyzer = self.get_analyzer()
            results = analyzer.analyze_paper(paper_text, analysis_options)
            
            # Add metadata about which provider was used
            results['ai_provider'] = self.get_current_provider()
            results['provider_model'] = getattr(analyzer, 'model_name', 'Unknown')
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed with {self.current_provider}: {e}")
            raise Exception(f"Analysis failed: {e}")
    
    def generate_flashcards(self, content: str) -> str:
        """Generate flashcards using current provider"""
        try:
            analyzer = self.get_analyzer()
            return analyzer.generate_flashcards(content)
        except Exception as e:
            logger.error(f"Flashcard generation failed with {self.current_provider}: {e}")
            raise Exception(f"Flashcard generation failed: {e}")
    
    def create_practice_questions(self, content: str, difficulty: str = "mixed") -> str:
        """Create practice questions using current provider"""
        try:
            analyzer = self.get_analyzer()
            return analyzer.create_practice_questions(content, difficulty)
        except Exception as e:
            logger.error(f"Practice questions creation failed with {self.current_provider}: {e}")
            raise Exception(f"Practice questions creation failed: {e}")
    
    def build_study_guide(self, content: str, focus_areas: list = None) -> str:
        """Build study guide using current provider"""
        try:
            analyzer = self.get_analyzer()
            return analyzer.build_study_guide(content, focus_areas)
        except Exception as e:
            logger.error(f"Study guide creation failed with {self.current_provider}: {e}")
            raise Exception(f"Study guide creation failed: {e}")
    
    def get_analyzer_info(self) -> Dict[str, Any]:
        """Get information about current analyzer"""
        try:
            analyzer = self.get_analyzer()
            return {
                'provider': self.get_current_provider(),
                'model_name': getattr(analyzer, 'model_name', 'Unknown'),
                'api_key_configured': hasattr(analyzer, 'api_key') and bool(analyzer.api_key),
                'status': 'Active'
            }
        except Exception as e:
            return {
                'provider': self.get_current_provider(),
                'model_name': 'Unknown',
                'api_key_configured': False,
                'status': f'Error: {e}'
            }

# Convenience function for easy initialization
def create_ai_analyzer(provider: str = "groq") -> UnifiedAIAnalyzer:
    """
    Create a unified AI analyzer with specified default provider.
    
    Args:
        provider: Default provider ("groq" or "gemini")
        
    Returns:
        UnifiedAIAnalyzer instance
    """
    return UnifiedAIAnalyzer(default_provider=provider)

# Example usage
if __name__ == "__main__":
    try:
        # Create unified analyzer
        ai_analyzer = create_ai_analyzer("groq")
        
        # Check available providers
        print("Available providers:", ai_analyzer.get_available_providers())
        print("Provider status:", ai_analyzer.get_provider_status())
        
        # Switch providers
        if ai_analyzer.switch_provider("gemini"):
            print("Successfully switched to Gemini")
        
        print("Current provider:", ai_analyzer.get_current_provider())
        print("Analyzer info:", ai_analyzer.get_analyzer_info())
        
    except Exception as e:
        print(f"Error: {e}")