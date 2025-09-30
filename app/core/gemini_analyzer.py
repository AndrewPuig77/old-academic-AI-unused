"""
Gemini AI Analyzer Module
Handles all interactions with Google's Gemini AI for research paper analysis
"""

import google.generativeai as genai
import os
import json
import logging
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiAnalyzer:
    def analyze_class_material_groq_style(self, content: str, material_type: str = "general") -> str:
        """Analyze class material for study purposes (Groq-style prompt)."""
        try:
            prompt = f"""
            Analyze this {material_type} class material to help students understand and learn effectively.

            🎓 CLASS MATERIAL ANALYSIS:

            **Content Overview**:
            - What is the main purpose of this material?
            - How does it fit into the broader course context?
            - What level of understanding is expected?

            **Learning Objectives**:
            - What should students be able to do after studying this?
            - Key skills and knowledge to be gained
            - Connection to course goals

            **Difficulty Assessment**:
            - What makes this material challenging?
            - Prerequisites needed
            - Estimated time required for mastery

            **Key Learning Points** (organized by priority):
            1. **Essential Concepts** - Must know
            2. **Important Details** - Should know
            3. **Supplementary Information** - Nice to know

            **Study Recommendations**:
            - Best approaches for this type of material
            - Effective study techniques
            - How to avoid common pitfalls

            **Assessment Preparation**:
            - Likely exam/assignment formats
            - Critical thinking questions
            - Practical applications

            **Connection Points**:
            - Links to other course materials
            - Real-world applications
            - Interdisciplinary connections

            Class Material: {content[:8000]}
            """
            return self._make_api_call_with_retry(prompt, operation_name="Groq-style class material analysis")
        except Exception as e:
            logger.error(f"Error analyzing class material (Groq-style): {e}")
            return f"Error analyzing class material: {str(e)}"
    def __init__(self, model_name: str = None):
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key or self.api_key == "your_google_api_key_here":
            raise ValueError("Please set GOOGLE_API_KEY in Streamlit secrets or .env file")
        genai.configure(api_key=self.api_key)
        # Dynamically select a working model
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            raise Exception("No Gemini models available for generateContent. Check your API key and quota.")
        # Prefer flash model if available
        flash_models = [m for m in available_models if 'flash' in m.lower()]
        self.model_name = model_name or (flash_models[0] if flash_models else available_models[0])
        self.model = genai.GenerativeModel(self.model_name)
        self.generation_config = {
            'temperature': float(os.getenv('GEMINI_TEMPERATURE', '0.1')),
            'max_output_tokens': int(os.getenv('GEMINI_MAX_TOKENS', '8192')),
            'top_p': 0.8,
            'top_k': 40
        }
        logger.info(f"Gemini analyzer initialized with model: {self.model_name}")
    
    def analyze_class_material(self, content: str, analysis_options: Dict[str, bool], material_type: str = "textbook") -> Dict[str, str]:
        """
        Analyze study material (textbook, lecture notes, handout, etc.) and return all requested analysis sections.
        """
        results = {}
        results['material_type'] = material_type
        results['content_length'] = len(content)
        try:
            # Summary
            if analysis_options.get('summary', False):
                results['summary'] = self._make_api_call_with_retry(
                    f"As an educational expert, provide a concise summary of the following {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="material summary"
                )
            # Key Concepts
            if analysis_options.get('concepts', False):
                results['concepts'] = self._make_api_call_with_retry(
                    f"List and explain the key concepts found in this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="key concepts extraction"
                )
            # Examples
            if analysis_options.get('examples', False):
                results['examples'] = self._make_api_call_with_retry(
                    f"Extract and describe important examples or case studies from this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="examples extraction"
                )
            # Keywords
            if analysis_options.get('keywords', False):
                results['keywords'] = self._make_api_call_with_retry(
                    self._get_keywords_prompt(content, material_type),
                    operation_name="keywords extraction"
                )
            # Detailed Analysis
            if analysis_options.get('detailed', False):
                results['detailed'] = self._make_api_call_with_retry(
                    f"Provide a detailed analysis of this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="detailed analysis"
                )
            # Questions
            if analysis_options.get('questions', False):
                results['questions'] = self.create_practice_questions(content)
            # Difficulty
            if analysis_options.get('difficulty', False):
                results['difficulty'] = self._make_api_call_with_retry(
                    f"Assess the difficulty level of this {material_type} material for students.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="difficulty assessment"
                )
            # Structure
            if analysis_options.get('structure', False):
                results['structure'] = self._make_api_call_with_retry(
                    f"Analyze the structure and organization of this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="structure analysis"
                )
            # Arguments
            if analysis_options.get('arguments', False):
                results['arguments'] = self._make_api_call_with_retry(
                    f"Identify and explain the key arguments presented in this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="arguments analysis"
                )
            # Improvements
            if analysis_options.get('improvements', False):
                results['improvements'] = self._make_api_call_with_retry(
                    f"Suggest improvements for this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="improvement suggestions"
                )
            # Main Points
            if analysis_options.get('main_points', False):
                results['main_points'] = self._make_api_call_with_retry(
                    f"List the main points covered in this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="main points extraction"
                )
            # Context
            if analysis_options.get('context', False):
                results['context'] = self._make_api_call_with_retry(
                    f"Analyze the context and background of this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="context analysis"
                )
            # Citations
            if analysis_options.get('citations', False):
                results['citations'] = self._make_api_call_with_retry(
                    f"Extract all citations and references from this {material_type} material.\n\nCONTENT:\n{content[:4000]}",
                    operation_name="citations extraction"
                )
            return results
        except Exception as e:
            logger.error(f"Error during study material analysis: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}")
        
    def _make_api_call_with_retry(self, prompt: str, max_retries: int = 5, operation_name: str = "API call") -> str:
        """
        Make API call with intelligent retry logic for quota errors.
        
        Args:
            prompt: The prompt to send to the model
            max_retries: Maximum number of retries (default 5)
            operation_name: Name of the operation for logging
            
        Returns:
            Generated text response
        """
        for attempt in range(max_retries + 1):
            try:
                response = self.model.generate_content(prompt, generation_config=self.generation_config)
                if response and response.text:
                    if attempt > 0:
                        logger.info(f"✅ {operation_name} succeeded on attempt {attempt + 1}")
                    return response.text
                else:
                    return f"⚠️ No response generated for {operation_name}"
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check for quota/rate limit errors (429 status)
                if ("429" in error_msg or "quota" in error_msg.lower() or 
                    "rate limit" in error_msg.lower() or "exceeded" in error_msg.lower()):
                    
                    if attempt < max_retries:
                        # Progressive backoff: 2, 5, 10, 20, 40 seconds
                        wait_time = min(2 ** (attempt + 1), 40)
                        logger.warning(f"🔄 {operation_name} - Quota exceeded, attempt {attempt + 1}/{max_retries + 1}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ {operation_name} failed after {max_retries + 1} attempts")
                        return f"⚠️ **API Quota Exceeded**\n\nThe Gemini API free tier has reached its daily limit. This is normal for free accounts.\n\n**Solutions:**\n• Wait 24 hours for quota reset\n• Upgrade to paid API plan for higher limits\n• Try using fewer analysis options at once\n\n**Note:** Your document was processed successfully, but AI analysis is temporarily limited."
                
                # Other errors (not quota related)
                else:
                    if attempt < max_retries:
                        wait_time = 2 * (attempt + 1)  # Simple backoff for other errors
                        logger.warning(f"🔄 {operation_name} error (attempt {attempt + 1}/{max_retries + 1}): {error_msg}. Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"❌ {operation_name} failed: {error_msg}")
                        return f"❌ **Error in {operation_name}**\n\n{error_msg}\n\nPlease try again or contact support if the issue persists."
        
        return f"❌ Failed to complete {operation_name} after {max_retries + 1} attempts"
    
    def analyze_paper(self, paper_text: str, analysis_options: Dict[str, bool]) -> Dict[str, str]:
        """
        Comprehensive analysis of academic content.
        
        Args:
            paper_text: Extracted text from the document
            analysis_options: Dictionary specifying which analyses to perform
            
        Returns:
            Dictionary containing analysis results
        """
        results = {}
        document_type = analysis_options.get('document_type', '📖 Other Academic Material')
        
        # Store document type in results for later use
        results['document_type'] = document_type
        
        try:
            # Generate summary if requested
            if analysis_options.get('summary', False):
                results['summary'] = self._make_api_call_with_retry(
                    self._get_summary_prompt(paper_text, document_type),
                    operation_name="summary generation"
                )
                time.sleep(1)  # Rate limiting
            
            # Research paper specific analyses
            if analysis_options.get('methodology', False):
                results['methodology'] = self._make_api_call_with_retry(
                    self._get_methodology_prompt(paper_text),
                    operation_name="methodology analysis"
                )
                time.sleep(1)
            
            if analysis_options.get('gaps', False):
                results['gaps'] = self._make_api_call_with_retry(
                    self._get_gaps_prompt(paper_text),
                    operation_name="research gaps identification"
                )
                time.sleep(1)
                
            if analysis_options.get('future_work', False):
                results['future_work'] = self._make_api_call_with_retry(
                    self._get_future_work_prompt(paper_text),
                    operation_name="future research suggestions"
                )
                time.sleep(1)
            
            # Additional analysis methods...
            if analysis_options.get('keywords', False):
                results['keywords'] = self._make_api_call_with_retry(
                    self._get_keywords_prompt(paper_text, document_type),
                    operation_name="keywords extraction"
                )
                time.sleep(1)
            
            # Citations
            if analysis_options.get('citations', False):
                results['citations'] = self._make_api_call_with_retry(
                    f"Extract all citations and references from this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="citations extraction"
                )
                time.sleep(1)
            # Concepts
            if analysis_options.get('concepts', False):
                results['concepts'] = self._make_api_call_with_retry(
                    f"List and explain the key concepts in this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="concepts extraction"
                )
                time.sleep(1)
            # Examples
            if analysis_options.get('examples', False):
                results['examples'] = self._make_api_call_with_retry(
                    f"Extract and describe important examples or case studies from this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="examples extraction"
                )
                time.sleep(1)
            # Questions
            if analysis_options.get('questions', False):
                results['questions'] = self.create_practice_questions(paper_text)
                time.sleep(1)
            # Difficulty
            if analysis_options.get('difficulty', False):
                results['difficulty'] = self._make_api_call_with_retry(
                    f"Assess the difficulty level of this document for students.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="difficulty assessment"
                )
                time.sleep(1)
            # Structure
            if analysis_options.get('structure', False):
                results['structure'] = self._make_api_call_with_retry(
                    f"Analyze the structure and organization of this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="structure analysis"
                )
                time.sleep(1)
            # Arguments
            if analysis_options.get('arguments', False):
                results['arguments'] = self._make_api_call_with_retry(
                    f"Identify and explain the key arguments presented in this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="arguments analysis"
                )
                time.sleep(1)
            # Improvements
            if analysis_options.get('improvements', False):
                results['improvements'] = self._make_api_call_with_retry(
                    f"Suggest improvements for this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="improvement suggestions"
                )
                time.sleep(1)
            # Findings
            if analysis_options.get('findings', False):
                results['findings'] = self._make_api_call_with_retry(
                    f"Summarize the key findings of this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="findings summary"
                )
                time.sleep(1)
            # Recommendations
            if analysis_options.get('recommendations', False):
                results['recommendations'] = self._make_api_call_with_retry(
                    f"Provide recommendations based on this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="recommendations"
                )
                time.sleep(1)
            # Main Points
            if analysis_options.get('main_points', False):
                results['main_points'] = self._make_api_call_with_retry(
                    f"List the main points covered in this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="main points extraction"
                )
                time.sleep(1)
            # Context
            if analysis_options.get('context', False):
                results['context'] = self._make_api_call_with_retry(
                    f"Analyze the context and background of this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="context analysis"
                )
                time.sleep(1)
            # Detailed Analysis
            if analysis_options.get('detailed', False):
                results['detailed'] = self._make_api_call_with_retry(
                    f"Provide a detailed analysis of this document.\n\nCONTENT:\n{paper_text[:4000]}",
                    operation_name="detailed analysis"
                )
                time.sleep(1)
            
            logger.info(f"Analysis completed with {len(results)} components")
            return results
            
        except Exception as e:
            logger.error(f"Error during document analysis: {str(e)}")
            raise Exception(f"Analysis failed: {str(e)}")
    
    def _get_summary_prompt(self, paper_text: str, document_type: str) -> str:
        """Generate summary prompt based on document type."""
        if "Research Paper" in document_type:
            return f"""
            As an expert research analyst, provide a comprehensive but concise summary of this research paper. 
            Include the following elements:

            📋 RESEARCH SUMMARY:
            • **Main Research Question/Problem**: What problem does this paper address?
            • **Key Methodology**: How did they approach the problem?
            • **Major Findings**: What are the most significant results?
            • **Practical Implications**: How can these findings be applied?
            • **Limitations**: What are the key limitations mentioned?

            Make the summary accessible to both experts and non-experts. Use clear, engaging language.
            Limit to 300-400 words.

            PAPER TEXT:
            {paper_text[:4000]}
            """
        else:
            return f"""
            As an educational content expert, provide a clear summary of this study material. 
            Focus on the learning objectives and key educational content:

            📚 CONTENT SUMMARY:
            • **Topic/Subject**: What is this content teaching?
            • **Key Concepts**: What are the main ideas or principles explained?
            • **Learning Objectives**: What should students understand after reading this?
            • **Practical Applications**: How are these concepts used in practice?
            • **Prerequisites**: What background knowledge is assumed?

            Write the summary in a student-friendly way that helps with understanding and retention.
            Limit to 300-400 words.

            STUDY MATERIAL TEXT:
            {paper_text[:4000]}
            """
    
    def _get_methodology_prompt(self, paper_text: str) -> str:
        """Generate methodology analysis prompt."""
        return f"""
        As a research methodology expert, analyze the research methods used in this paper. 
        Provide a detailed breakdown:

        🔬 METHODOLOGY ANALYSIS:
        • **Research Design**: What type of study is this? (experimental, observational, theoretical, etc.)
        • **Data Collection**: How was data gathered? What instruments/tools were used?
        • **Sample/Participants**: Who or what was studied? Sample size and characteristics?
        • **Analysis Methods**: What statistical or analytical methods were employed?
        • **Variables**: What were the key independent and dependent variables?
        • **Controls**: What controls or comparisons were made?
        • **Validity Considerations**: How did they ensure validity and reliability?

        Rate the methodology strength from 1-5 and explain why.
        Suggest improvements or alternative approaches.

        PAPER TEXT:
        {paper_text[:4000]}
        """
    
    def _get_gaps_prompt(self, paper_text: str) -> str:
        """Generate research gaps identification prompt."""
        return f"""
        As a research strategist, identify research gaps and future opportunities based on this paper:

        🔍 RESEARCH GAPS & OPPORTUNITIES:
        • **Explicit Gaps**: What gaps do the authors explicitly mention?
        • **Implicit Gaps**: What gaps can you infer from the methodology or findings?
        • **Methodological Improvements**: What methodological enhancements could strengthen future research?
        • **Scale & Scope**: Could the research be expanded in scale, scope, or context?
        • **Interdisciplinary Opportunities**: What other fields could contribute to or benefit from this research?
        • **Practical Applications**: What real-world applications need further development?
        • **Replication Needs**: What aspects need validation through replication?

        Prioritize the gaps by potential impact and feasibility.
        Suggest specific research questions for future studies.

        PAPER TEXT:
        {paper_text[:4000]}
        """
    
    def _get_future_work_prompt(self, paper_text: str) -> str:
        """Generate future research suggestions prompt."""
        return f"""
        As a research director, suggest specific future research directions based on this work:

        🔮 FUTURE RESEARCH DIRECTIONS:
        • **Immediate Next Steps**: Short-term research opportunities building directly on this work
        • **Long-term Investigations**: Major research programs that could emerge from these findings
        • **Interdisciplinary Connections**: How other fields could contribute to or benefit from this research
        • **Methodological Advances**: New approaches, tools, or techniques that could be developed
        • **Practical Applications**: Real-world implementation studies and pilot programs
        • **Scaling Studies**: Research on broader populations, contexts, or applications

        Focus on feasible and impactful research directions with clear value propositions.

        PAPER TEXT:
        {paper_text[:4000]}
        """
    
    def _get_keywords_prompt(self, paper_text: str, document_type: str) -> str:
        """Generate keywords extraction prompt."""
        if "Research Paper" in document_type:
            return f"""
            As a domain expert, extract research terminology and keywords:

            🏷️ RESEARCH KEYWORDS:
            • **Primary Keywords**: Core research terms that define this work
            • **Technical Terminology**: Specialized vocabulary and jargon
            • **Theoretical Concepts**: Key theoretical terms and frameworks
            • **Methodological Terms**: Research method vocabulary
            • **Field-Specific Language**: Discipline-specific terms and concepts

            Focus on terms crucial for research discovery and academic discourse.
            Format as a well-organized list with brief explanations where helpful.

            PAPER TEXT:
            {paper_text[:4000]}
            """
        else:
            return f"""
            As an educational vocabulary expert, extract key terms and concepts:

            🏷️ KEY TERMS & VOCABULARY:
            • **Essential Terms**: Must-know vocabulary for understanding this material
            • **Concept Names**: Important concept labels and terminology
            • **Technical Terms**: Specialized vocabulary explained in the content
            • **Study Terms**: Terms students should memorize and understand
            • **Context Usage**: How terms are used and defined in this context

            Focus on vocabulary important for learning and comprehension.
            Format as a study-friendly glossary.

            STUDY MATERIAL TEXT:
            {paper_text[:4000]}
            """
    
    def generate_flashcards(self, content: str) -> str:
        """Generate flashcards for study purposes."""
        try:
            prompt = f"""
            Create 15-20 high-quality flashcards based on this academic content for effective studying.

            🃏 FLASHCARD CREATION GUIDELINES:

            **Format Requirements**:
            - Use clear section headers for organization
            - Number each flashcard clearly
            - Separate FRONT and BACK content distinctly
            - Make content easy to read and study from

            **Structure each flashcard as**:
            ---
            **FLASHCARD #[number]: [Card Type]**
            
            **FRONT:** [Clear question or term]
            
            **BACK:** [Complete, concise answer with key details]
            ---

            **Types of flashcards to create**:
            1. **Definition Cards** (6-8 cards): Key terms and their precise definitions
            2. **Concept Cards** (4-5 cards): Important concepts and explanations  
            3. **Process Cards** (2-3 cards): Steps, procedures, or methodologies
            4. **Comparison Cards** (2-3 cards): Contrasting ideas or approaches
            5. **Application Cards** (2-3 cards): Examples and real-world applications

            **Quality Standards**:
            - FRONT: Clear, specific question or term that tests understanding
            - BACK: Complete but concise answer (2-4 sentences maximum)
            - Avoid yes/no questions - use "What is...?", "How does...?", "Why...?"
            - Test one focused concept per card
            - Use active recall principles for maximum learning effectiveness
            - Include context when helpful for understanding

            Content to analyze: {content[:8000]}
            """
            
            return self._make_api_call_with_retry(prompt, operation_name="flashcard generation")
            
        except Exception as e:
            logger.error(f"Error generating flashcards: {e}")
            return f"Error generating flashcards: {str(e)}"

    def create_practice_questions(self, content: str, difficulty: str = "mixed") -> str:
        """Create practice questions for study purposes."""
        try:
            prompt = f"""
            Create comprehensive practice questions at {difficulty} difficulty level based on this content.

            📝 PRACTICE QUESTIONS SET:

            **Question Types** (20-25 total questions):

            **A. Multiple Choice** (8-10 questions):
            - Include 4 options (a, b, c, d)
            - One clearly correct answer
            - Plausible distractors
            - Test both knowledge and understanding

            **B. Short Answer** (6-8 questions):
            - Require 2-3 sentence responses
            - Test comprehension and analysis
            - Clear, specific questions

            **C. Essay Questions** (3-4 questions):
            - Require detailed responses
            - Test critical thinking and synthesis
            - Include specific instructions

            **D. Application Questions** (3-5 questions):
            - Present scenarios or cases
            - Require applying concepts
            - Test practical understanding

            **Answer Key**:
            Provide complete answers for all questions, including:
            - Correct options for multiple choice
            - Key points for short answers
            - Detailed outlines for essays

            Content: {content[:8000]}
            """
            
            return self._make_api_call_with_retry(prompt, operation_name="practice questions creation")
            
        except Exception as e:
            logger.error(f"Error creating practice questions: {e}")
            return f"Error creating practice questions: {str(e)}"

    def build_study_guide(self, content: str, focus_areas: list = None) -> str:
        """Build a comprehensive study guide."""
        try:
            focus_text = f" with emphasis on: {', '.join(focus_areas)}" if focus_areas else ""
            prompt = f"""
            Create a comprehensive study guide{focus_text} based on this academic content.

            📚 COMPREHENSIVE STUDY GUIDE:

            **I. Executive Summary** (150 words)
            - Key takeaways and main points
            - Why this material is important

            **II. Core Concepts & Definitions** 
            - 10-15 key terms with clear definitions
            - Organized by importance and relationships

            **III. Main Topics Breakdown**
            For each major topic:
            - Overview and significance
            - Key points and sub-concepts
            - Relationships to other topics
            - Common misconceptions

            **IV. Visual Learning Aids**
            - Concept maps or hierarchies (described in text)
            - Process flows or timelines
            - Comparison tables

            **V. Study Strategies**
            - How to approach this material
            - Connection points between concepts
            - Practical applications

            **VI. Self-Assessment Tools**
            - Key questions to test understanding
            - Red flags/common mistakes
            - Study progress checkpoints

            Academic material: {content[:8000]}
            """
            
            return self._make_api_call_with_retry(prompt, operation_name="study guide creation")
            
        except Exception as e:
            logger.error(f"Error building study guide: {e}")
            return f"Error building study guide: {str(e)}"

    def suggest_related_papers(self, content: str) -> str:
        """Suggest related research papers based on content."""
        try:
            prompt = f"""
            Suggest 5-10 highly relevant research papers (with titles, authors, and publication years) based on the following content. Focus on recent, high-impact, and closely related work. If possible, include links or DOIs.

            CONTENT:
            {content[:4000]}
            """
            return self._make_api_call_with_retry(prompt, operation_name="related papers suggestion")
        except Exception as e:
            logger.error(f"Error suggesting related papers: {e}")
            return f"Error suggesting related papers: {str(e)}"

    def generate_research_questions(self, content: str) -> str:
        """Generate research questions based on content."""
        try:
            prompt = f"""
            Generate 5-10 advanced research questions inspired by the following content. Focus on open problems, future directions, and gaps in the field.

            CONTENT:
            {content[:4000]}
            """
            return self._make_api_call_with_retry(prompt, operation_name="research questions generation")
        except Exception as e:
            logger.error(f"Error generating research questions: {e}")
            return f"Error generating research questions: {str(e)}"

    def build_hypotheses(self, content: str) -> str:
        """Build hypotheses based on content."""
        try:
            prompt = f"""
            Formulate 3-5 testable hypotheses based on the following content. Each hypothesis should be clear, specific, and grounded in the material.

            CONTENT:
            {content[:4000]}
            """
            return self._make_api_call_with_retry(prompt, operation_name="hypotheses building")
        except Exception as e:
            logger.error(f"Error building hypotheses: {e}")
            return f"Error building hypotheses: {str(e)}"

    def generate_research_proposal(self, content: str) -> str:
        """Generate a research proposal based on content."""
        try:
            prompt = f"""
            Draft a 1-2 page research proposal based on the following content. Include background, objectives, methodology, expected outcomes, and significance.

            CONTENT:
            {content[:4000]}
            """
            return self._make_api_call_with_retry(prompt, operation_name="research proposal generation")
        except Exception as e:
            logger.error(f"Error generating research proposal: {e}")
            return f"Error generating research proposal: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    try:
        analyzer = GeminiAnalyzer()
        print("Gemini Analyzer initialized successfully!")
        
        # Test with sample text
        sample_text = "This is a test research paper about artificial intelligence..."
        test_options = {'summary': True, 'keywords': True}
        
        # This would be used for testing
        # results = analyzer.analyze_paper(sample_text, test_options)
        # print("Test analysis completed!")
        
    except Exception as e:
        print(f"Error: {e}")