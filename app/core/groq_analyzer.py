"""
Groq AI Analyzer Module
Handles all interactions with Groq's LLM API for research paper analysis
Free tier: 14,400 requests per day - Perfect for academic use!
"""

import groq
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

class GroqAnalyzer:
    """
    Handles analysis of research papers using Groq's LLM API.
    Provides specialized prompts and analysis functions for academic content.
    Uses free tier with generous 14,400 requests/day limit.
    """
    
    def __init__(self, model_name: str = "llama-3.1-8b-instant"):
        """
        Initialize Groq analyzer with API configuration.
        
        Args:
            model_name: Name of the Groq model to use
        """
        self.model_name = model_name
        
        # Try to get API key from Streamlit secrets first, then environment
        self.api_key = None
        try:
            import streamlit as st
            self.api_key = st.secrets.get("GROQ_API_KEY")
        except:
            self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            raise ValueError("Please set GROQ_API_KEY in Streamlit secrets or .env file")
        
        # Initialize Groq client
        self.client = groq.Groq(api_key=self.api_key)
        
        # Test the connection
        try:
            test_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.1
            )
            if test_response:
                logger.info(f"Successfully initialized Groq model: {self.model_name}")
            else:
                raise Exception("No response from Groq API")
        except Exception as e:
            logger.error(f"Failed to initialize Groq model: {e}")
            raise ValueError(f"Could not initialize Groq model: {e}")
    
    def _get_document_description(self, document_type: str) -> dict:
        """
        Get appropriate terminology for different document types.
        
        Args:
            document_type: The document type from the UI selection
            
        Returns:
            Dictionary with appropriate terms for the document type
        """
        type_mapping = {
            '🔬 Research Paper': {
                'name': 'research paper',
                'action': 'research paper analysis',
                'context': 'academic research'
            },
            '📚 Textbook Chapter': {
                'name': 'textbook chapter', 
                'action': 'educational content analysis',
                'context': 'educational material'
            },
            '📝 Lecture Notes': {
                'name': 'lecture notes',
                'action': 'lecture content analysis', 
                'context': 'educational content'
            },
            '📋 Assignment/Homework': {
                'name': 'assignment material',
                'action': 'assignment analysis',
                'context': 'educational assignment'
            },
            '📄 Article/Essay': {
                'name': 'article',
                'action': 'article analysis',
                'context': 'written article'
            },
            '📊 Report/Thesis': {
                'name': 'academic report',
                'action': 'report analysis',
                'context': 'academic document'
            },
            '🎓 Study Guide': {
                'name': 'study guide',
                'action': 'study material analysis',
                'context': 'study material'
            },
            '🗒️ Class Handout': {
                'name': 'class handout',
                'action': 'handout analysis',
                'context': 'educational material'
            },
            '📖 Other Academic Material': {
                'name': 'academic document',
                'action': 'document analysis',
                'context': 'academic material'
            }
        }
        
        return type_mapping.get(document_type, {
            'name': 'academic document',
            'action': 'document analysis', 
            'context': 'academic material'
        })
    
    def analyze_research_paper(self, text: str, analysis_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Analyze research paper content using Groq AI.
        
        Args:
            text: The research paper text to analyze
            analysis_type: Type of analysis to perform
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            prompt = self._get_analysis_prompt(text, analysis_type)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.1
            )
            
            if response and response.choices:
                # Parse the response
                analysis_text = response.choices[0].message.content
                return self._parse_analysis_response(analysis_text, analysis_type)
            else:
                return {"error": "No response from Groq API"}
                
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}
    
    def generate_summary(self, text: str, summary_type: str = "abstract") -> Dict[str, Any]:
        """
        Generate summary of research paper using Groq AI.
        
        Args:
            text: The research paper text to summarize
            summary_type: Type of summary to generate
            
        Returns:
            Dictionary containing summary results
        """
        try:
            prompt = self._get_summary_prompt(text, summary_type)
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1
            )
            
            if response and response.choices:
                summary_text = response.choices[0].message.content
                return {
                    "summary": summary_text,
                    "type": summary_type,
                    "word_count": len(summary_text.split())
                }
            else:
                return {"error": "No response from Groq API"}
                
        except Exception as e:
            logger.error(f"Error during summarization: {e}")
            return {"error": f"Summarization failed: {str(e)}"}
    
    def extract_key_information(self, text: str) -> Dict[str, Any]:
        """
        Extract key information from research paper using Groq AI.
        
        Args:
            text: The research paper text to analyze
            
        Returns:
            Dictionary containing extracted information
        """
        try:
            prompt = f"""
            Please extract the following key information from this research paper:

            1. **Title**: The main title of the paper
            2. **Authors**: List of authors
            3. **Abstract**: The abstract/summary section
            4. **Keywords**: Important keywords or terms
            5. **Research Question**: Main research question or hypothesis
            6. **Methodology**: Research methods used
            7. **Key Findings**: Main results or findings
            8. **Conclusions**: Primary conclusions
            9. **Limitations**: Any mentioned limitations
            10. **Future Work**: Suggested future research directions

            Please format your response as a JSON object with these fields.

            Research Paper Text:
            {text[:12000]}  # Limit text to avoid token limits
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.1
            )
            
            if response and response.choices:
                extraction_text = response.choices[0].message.content
                try:
                    # Try to parse as JSON
                    extracted_info = json.loads(extraction_text)
                    return extracted_info
                except json.JSONDecodeError:
                    # If not valid JSON, return as structured text
                    return {"extracted_info": extraction_text}
            else:
                return {"error": "No response from Groq API"}
                
        except Exception as e:
            logger.error(f"Error during key information extraction: {e}")
            return {"error": f"Extraction failed: {str(e)}"}
    
    def generate_study_questions(self, text: str, difficulty: str = "intermediate") -> Dict[str, Any]:
        """
        Generate study questions from research paper using Groq AI.
        
        Args:
            text: The research paper text
            difficulty: Difficulty level (basic, intermediate, advanced)
            
        Returns:
            Dictionary containing generated questions
        """
        try:
            prompt = f"""
            Generate study questions for this research paper at {difficulty} level.
            Include different types of questions:
            
            1. **Comprehension Questions** (5 questions) - Test understanding of main concepts
            2. **Analysis Questions** (5 questions) - Require deeper thinking about methods and findings
            3. **Critical Thinking Questions** (3 questions) - Evaluate and critique the research
            4. **Application Questions** (3 questions) - Apply concepts to new scenarios
            
            Format as a JSON object with these categories as keys, each containing an array of questions.

            Research Paper Text:
            {text[:10000]}  # Limit text to avoid token limits
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.3
            )
            
            if response and response.choices:
                questions_text = response.choices[0].message.content
                try:
                    questions = json.loads(questions_text)
                    return questions
                except json.JSONDecodeError:
                    return {"study_questions": questions_text}
            else:
                return {"error": "No response from Groq API"}
                
        except Exception as e:
            logger.error(f"Error during question generation: {e}")
            return {"error": f"Question generation failed: {str(e)}"}
    
    def _get_analysis_prompt(self, text: str, analysis_type: str) -> str:
        """Get the appropriate prompt for analysis type."""
        
        if analysis_type == "comprehensive":
            return f"""
            Please provide a comprehensive analysis of this research paper. Include:

            1. **Overview**: Brief summary of the paper's purpose and scope
            2. **Research Question/Hypothesis**: What the authors are trying to answer or prove
            3. **Methodology**: How the research was conducted
            4. **Key Findings**: Main results and discoveries
            5. **Strengths**: What the paper does well
            6. **Weaknesses/Limitations**: Areas where the paper could be improved
            7. **Significance**: Why this research matters to the field
            8. **Future Directions**: What research this suggests for the future

            Research Paper:
            {text[:12000]}
            """
        
        elif analysis_type == "methodology":
            return f"""
            Please analyze the methodology of this research paper. Focus on:

            1. **Research Design**: What type of study/experiment is this?
            2. **Data Collection**: How was data gathered?
            3. **Sample Size**: How many participants/subjects?
            4. **Variables**: Independent and dependent variables
            5. **Controls**: What controls were used?
            6. **Statistical Analysis**: What statistical methods were employed?
            7. **Validity**: Internal and external validity considerations
            8. **Reproducibility**: Can this study be replicated?

            Research Paper:
            {text[:12000]}
            """
        
        else:  # Default comprehensive
            return self._get_analysis_prompt(text, "comprehensive")
    
    def _get_summary_prompt(self, text: str, summary_type: str) -> str:
        """Get the appropriate prompt for summary type."""
        
        if summary_type == "abstract":
            return f"""
            Create a concise abstract-style summary of this research paper (150-250 words).
            Include: purpose, methods, key findings, and conclusions.

            Research Paper:
            {text[:10000]}
            """
        
        elif summary_type == "executive":
            return f"""
            Create an executive summary for this research paper (300-500 words).
            Focus on practical implications and key takeaways for decision makers.

            Research Paper:
            {text[:10000]}
            """
        
        else:  # Default to abstract
            return self._get_summary_prompt(text, "abstract")
    
    def _parse_analysis_response(self, response_text: str, analysis_type: str) -> Dict[str, Any]:
        """Parse Groq's response into structured format."""
        try:
            # Try to extract structured sections from the response
            sections = {}
            current_section = "general"
            current_content = []
            
            for line in response_text.split('\n'):
                line = line.strip()
                if line.startswith('**') and line.endswith('**'):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = line.strip('**').lower().replace(' ', '_')
                    current_content = []
                elif line:
                    current_content.append(line)
            
            # Save final section
            if current_content:
                sections[current_section] = '\n'.join(current_content)
            
            return {
                "analysis_type": analysis_type,
                "full_analysis": response_text,
                "sections": sections,
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.warning(f"Could not parse response into sections: {e}")
            return {
                "analysis_type": analysis_type,
                "full_analysis": response_text,
                "timestamp": time.time()
            }
    
    def analyze_paper(self, paper_text: str, analysis_options: Dict[str, bool]) -> Dict[str, str]:
        """
        Comprehensive analysis of academic content with sophisticated prompting.
        
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
            # Get document-specific terminology
            doc_info = self._get_document_description(document_type)
            doc_name = doc_info['name']
            doc_context = doc_info['context']
            
            # Generate sophisticated summary if requested
            if analysis_options.get('summary', False):
                summary_prompt = f"""
                As an expert academic analyst, create a comprehensive, student-friendly analysis of this {doc_name}. Format your response with clear structure and educational value.

                ## � EXECUTIVE SUMMARY

                **Topic/Subject**: Clearly identify the main topic or subject matter. What specific concepts, theories, or areas does this {doc_name} address?

                **Key Concepts**: List and explain the 3-5 most important concepts, theories, or ideas presented. For each concept:
                - Provide a clear definition
                - Explain its significance
                - Give practical context

                **Learning Objectives**: What should students be able to understand or do after studying this material? List 4-6 specific learning outcomes.

                **Content Structure**: How is the information organized? What approach does the author take to present the material?

                **Practical Applications**: How can this knowledge be applied in real-world situations? What are the practical implications for:
                - Students in their studies
                - Professionals in the field  
                - Future learning and development

                **Prerequisites**: What background knowledge would help students better understand this material?

                **Key Takeaways**: What are the 3-4 most essential points students should remember from this {doc_name}?

                Format with clear headings, bullet points, and educational language. Make it comprehensive but accessible to students.

                {doc_name.upper()} TEXT:
                {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": summary_prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                results['summary'] = response.choices[0].message.content
                time.sleep(0.5)
            
            # Sophisticated methodology analysis
            if analysis_options.get('methodology', False):
                # Adjust methodology prompt based on document type
                if document_type == '🔬 Research Paper':
                    methodology_prompt = f"""
                    As a methodological expert and research design specialist, provide a comprehensive analysis of this research paper's methodology. Your analysis should demonstrate deep understanding of research design principles and methodological rigor.

                    🔬 METHODOLOGY ANALYSIS:

                    **Research Design**: What type of study/experiment is this? Evaluate the appropriateness of the design for addressing the research question.
                    **Data Collection**: Analyze the data collection methods, instruments, and procedures used.
                    **Sample/Participants**: Examine the sample characteristics, size, and selection methods.
                    **Variables and Measurements**: Identify and analyze key variables and how they were measured.
                    **Statistical Analysis**: Evaluate the statistical methods used and their appropriateness.
                    **Validity and Limitations**: Assess internal/external validity and methodological limitations.

                    {doc_name.upper()} TEXT:
                    {paper_text[:8000]}
                    """
                else:
                    methodology_prompt = f"""
                    As an educational content expert, provide a comprehensive analysis of how this {doc_name} presents and organizes information. Your analysis should focus on pedagogical approach and content structure.

                    📚 CONTENT APPROACH ANALYSIS:

                    **Content Organization**: How is the information structured and organized? What pedagogical approach is used?
                    **Learning Objectives**: What key concepts or skills does this material aim to teach?
                    **Presentation Methods**: What methods are used to present information (examples, diagrams, exercises, etc.)?
                    **Difficulty Level**: What is the appropriate academic level for this content?
                    **Educational Value**: How effective is this material for learning and understanding the subject?
                    **Key Strengths**: What makes this {doc_name} particularly effective or valuable?

                    {doc_name.upper()} TEXT:
                    {paper_text[:8000]}
                    """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": methodology_prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                results['methodology'] = response.choices[0].message.content
                time.sleep(0.5)

            # Research gaps analysis
            if analysis_options.get('gaps', False):
                gaps_prompt = f"""
                As a research strategist with deep expertise in identifying research opportunities, analyze this paper to identify significant research gaps and future research directions. Your analysis should demonstrate sophisticated understanding of the research landscape.

                🔍 RESEARCH GAPS IDENTIFIED:

                **Explicit Gaps**: What gaps does the paper explicitly acknowledge? Are these well-justified?

                **Implicit Gaps**: What important questions or limitations become apparent from your analysis that the authors may not have fully addressed?

                **Methodological Gaps**: What methodological improvements or alternative approaches could strengthen this research area?

                **Scale & Scope Gaps**: 
                - Geographic or demographic limitations that could be addressed
                - Temporal aspects that need investigation
                - Sample size or population coverage issues

                **Interdisciplinary Opportunities**: How could insights from other disciplines enhance this research? What collaborative opportunities exist?

                **Technological Gaps**: How could emerging technologies or methods advance this research area?

                **Practical Applications**: What gaps exist between theoretical findings and practical implementation?

                **Prioritized Research Questions**: 
                Provide 5-7 specific, well-formulated research questions that would address the most important gaps, ranked by:
                1. Scientific impact potential
                2. Feasibility
                3. Societal relevance

                For each prioritized question, briefly explain:
                - Why it's important
                - What methodology would be most appropriate
                - What resources/expertise would be required

                **Future Research Roadmap**: Outline a logical sequence of research studies that could systematically address these gaps over the next 5-10 years.

                Demonstrate strategic thinking and deep domain expertise. Show how addressing these gaps could advance the field significantly.

                RESEARCH PAPER TEXT:
                {paper_text[:10000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": gaps_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['gaps'] = response.choices[0].message.content
                time.sleep(0.5)
            
            # Keywords and concepts extraction
            if analysis_options.get('keywords', False):
                keywords_prompt = f"""
                As an expert educator and content analyst, extract and categorize all key terms and concepts from this {doc_name}. Format your response like a comprehensive study vocabulary list.

                ## 🏷️ KEY TERMS & CONCEPTS

                **Essential Terms**: List and define the 10-15 most important terms that students absolutely must understand. For each term, provide:
                - Clear, student-friendly definition
                - Context of how it's used in the material
                - Why it's important to understand

                **Concept Names**: Identify 8-12 specific concepts, theories, or frameworks mentioned. Explain:
                - What each concept represents
                - How it relates to the main topic
                - Any examples or applications provided

                **Technical Terms**: List 10-15 specialized vocabulary terms with:
                - Precise definitions
                - Technical context
                - Related terms or synonyms

                **Study Terms**: Identify terms that are crucial for:
                - Exams and assessments
                - Further learning in this subject
                - Practical application

                **Context Clues**: For complex terms, provide:
                - Examples from the text that help understand the meaning
                - Related terms that provide context
                - Common misconceptions to avoid

                **Acronyms and Abbreviations**: List any acronyms with full expansions and explanations.

                **Cross-References**: Identify terms that connect to:
                - Other chapters or sections
                - Related subjects or courses
                - Real-world applications

                Format each section clearly with bullet points and comprehensive explanations that would help students create effective study materials.

                {doc_name.upper()} TEXT:
                {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": keywords_prompt}],
                    max_tokens=1500,
                    temperature=0.1
                )
                results['keywords'] = response.choices[0].message.content
                time.sleep(0.5)
            
            # Generate comprehensive study questions
            if analysis_options.get('questions', False):
                questions_prompt = f"""
                As an expert educator and assessment specialist, generate sophisticated study questions that test deep understanding of this research paper at multiple cognitive levels.

                📚 COMPREHENSIVE STUDY QUESTIONS:

                **Comprehension Questions** (5 questions):
                Test basic understanding of key concepts, findings, and methodology. These should ensure students can accurately summarize and explain the core elements of the research.

                **Analysis Questions** (5 questions):
                Require students to break down complex concepts, compare different elements, and examine relationships between components of the research.

                **Critical Evaluation Questions** (4 questions):
                Challenge students to assess the quality, validity, and significance of the research. Include questions about methodology critique and limitation analysis.

                **Synthesis Questions** (3 questions):
                Ask students to combine information from this research with other knowledge to create new insights or propose novel applications.

                **Application Questions** (4 questions):
                Require students to apply the research findings to new scenarios, problems, or contexts not directly addressed in the paper.

                **Research Design Questions** (3 questions):
                Challenge students to design follow-up studies, propose methodological improvements, or suggest alternative approaches.

                For each question:
                - Provide clear, specific wording
                - Include any necessary context
                - Suggest key points that should be addressed in a complete answer
                - Indicate approximate difficulty level and time requirement

                **Answer Keys**: For comprehension and analysis questions, provide brief outline answers highlighting the key points students should address.

                Ensure questions are intellectually rigorous while being clear and fair. Avoid questions that can be answered with simple recall.

                RESEARCH PAPER TEXT:
                {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": questions_prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                results['questions'] = response.choices[0].message.content
                time.sleep(0.5)
            
            # Citations and references analysis
            if analysis_options.get('citations', False):
                citations_prompt = f"""
                As a bibliometric expert and research librarian, analyze the citations and references in this research paper to evaluate the scholarly foundation and identify key literature.

                📚 CITATIONS & REFERENCES ANALYSIS:

                **Reference Quality Assessment**:
                - How comprehensive is the literature review?
                - Are key seminal works in the field included?
                - Is the citation coverage balanced across different perspectives?
                - Are there notable omissions of important research?

                **Citation Currency**:
                - What is the age distribution of cited works?
                - Are recent developments in the field adequately represented?
                - Is there an appropriate balance of classic and current sources?

                **Source Diversity**:
                - Types of sources cited (journal articles, books, reports, etc.)
                - Geographic and institutional diversity of cited authors
                - Methodological diversity in cited studies

                **Key Citations Identified**:
                Extract and categorize the most important citations:
                - **Foundational Works**: Seminal papers that established key concepts
                - **Methodological Sources**: Papers that informed the research approach
                - **Comparative Studies**: Research with similar aims or methods
                - **Contradictory Evidence**: Papers that present conflicting findings

                **Citation Analysis**:
                - Average citations per page/1000 words
                - Self-citation patterns (if apparent)
                - Citation recency (% of citations from last 5 years)
                - Journal quality indicators for major citations

                **Literature Gaps**:
                - What important research areas seem underrepresented?
                - Are there geographic or demographic biases in the literature cited?
                - What emerging research directions are not adequately covered?

                **Recommended Additional Sources**:
                Suggest 5-10 additional key papers that would strengthen the literature foundation, with brief justification for each.

                **Citation Network Insights**:
                - Which authors or research groups appear most influential?
                - What are the main research clusters or schools of thought?
                - How does this paper position itself within the citation network?

                Focus on extracting actual citations and references that appear in the paper. Be specific about authors, titles, and years when possible.

                RESEARCH PAPER TEXT:
                {paper_text[:12000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": citations_prompt}],
                    max_tokens=2000,
                    temperature=0.1
                )
                results['citations'] = response.choices[0].message.content
                time.sleep(0.5)
            
            # Future research directions analysis
            if analysis_options.get('future_work', False):
                future_work_prompt = f"""
                As a strategic research visionary and academic futurist, identify and elaborate on future research directions that emerge from this paper's findings and limitations.

                🔮 FUTURE RESEARCH DIRECTIONS:

                **Immediate Next Steps** (1-2 years):
                - What are the most logical and feasible follow-up studies?
                - Which specific limitations could be addressed quickly?
                - What methodological improvements could be implemented immediately?
                - Which findings need replication or validation?

                **Medium-Term Research Agenda** (3-5 years):
                - What broader questions emerge from these findings?
                - How could this research be scaled or extended?
                - What interdisciplinary collaborations would be valuable?
                - Which theoretical frameworks need development?

                **Long-Term Vision** (5-10 years):
                - What paradigm shifts might this research contribute to?
                - How could emerging technologies transform this research area?
                - What societal challenges could this research help address?
                - Which fundamental questions remain unanswered?

                **Specific Research Questions for Future Investigation**:
                Generate 8-10 specific, actionable research questions organized by:
                1. **Mechanistic Questions**: How and why do these phenomena occur?
                2. **Boundary Conditions**: When and where do these effects apply?
                3. **Causal Questions**: What drives the relationships observed?
                4. **Application Questions**: How can these findings be practically applied?
                5. **Scale Questions**: Do these findings hold at different levels of analysis?

                **Methodological Innovations Needed**:
                - What new research methods would advance this field?
                - Which measurement or analytical approaches need development?
                - How could technology improve data collection or analysis?
                - What collaborative frameworks would enhance research quality?

                **Interdisciplinary Opportunities**:
                - Which fields could contribute valuable perspectives?
                - What cross-disciplinary methodologies could be applied?
                - Where might unexpected connections lead to breakthroughs?

                **Resource and Infrastructure Requirements**:
                - What funding initiatives would accelerate progress?
                - Which research infrastructure developments are needed?
                - What training or capacity-building is required?

                **Potential Impact and Applications**:
                - How might this research influence policy or practice?
                - What industries or sectors could benefit?
                - Which social challenges could be addressed?

                Prioritize directions by potential impact, feasibility, and scientific importance. Be visionary but grounded in realistic possibilities.

                RESEARCH PAPER TEXT:
                {paper_text[:10000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": future_work_prompt}],
                    max_tokens=2500,
                    temperature=0.2
                )
                results['future_work'] = response.choices[0].message.content
                time.sleep(0.5)
            
            # Key concepts analysis (for study materials)
            if analysis_options.get('concepts', False):
                concepts_prompt = f"""
                As an educational expert and curriculum specialist, identify and explain the key concepts from this academic material in a way that facilitates deep learning.

                🎯 KEY CONCEPTS ANALYSIS:

                **Core Concepts** (5-8 primary concepts):
                For each concept, provide:
                - **Clear Definition**: Precise, accessible explanation
                - **Significance**: Why this concept is important
                - **Context**: How it fits within the broader subject
                - **Prerequisites**: What students need to know first

                **Supporting Concepts** (8-12 secondary concepts):
                - Terms and ideas that support the core concepts
                - Brief explanations and connections
                - Relationship to core concepts

                **Conceptual Hierarchy**:
                - Organize concepts from fundamental to advanced
                - Show relationships and dependencies
                - Create a logical learning progression

                **Learning Objectives**:
                - What students should understand about each concept
                - Skills they should develop
                - Applications they should be able to make

                **Common Misconceptions**:
                - Typical errors students make with these concepts
                - Clarifications and corrections
                - Ways to avoid confusion

                **Memory Aids**:
                - Mnemonics or memory techniques
                - Analogies that help explain difficult concepts
                - Visual or spatial learning connections

                **Assessment Questions**:
                - Questions to test concept understanding
                - Different levels of cognitive demand
                - Real-world applications

                Academic Material: {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": concepts_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['concepts'] = response.choices[0].message.content
                time.sleep(0.5)

            # Examples and cases analysis (for study materials)
            if analysis_options.get('examples', False):
                examples_prompt = f"""
                As an educational designer and case study expert, extract and analyze examples, cases, and applications from this academic material to enhance learning.

                💡 EXAMPLES & CASES ANALYSIS:

                **Primary Examples** (5-8 detailed examples):
                For each example:
                - **Description**: What the example demonstrates
                - **Concept Connection**: Which concepts it illustrates
                - **Analysis**: Why this example is effective
                - **Learning Value**: What students gain from it

                **Case Studies** (if present):
                - Detailed analysis of any case studies
                - Key lessons and takeaways
                - Broader applications and implications
                - Questions for deeper analysis

                **Real-World Applications**:
                - How concepts apply in practice
                - Industry or professional contexts
                - Contemporary relevance
                - Career connections

                **Comparative Examples**:
                - Examples that show contrasts or variations
                - What makes examples different or similar
                - When to apply different approaches

                **Student-Generated Examples**:
                - Suggestions for examples students could create
                - Templates for applying concepts
                - Practice scenarios

                **Visual and Concrete Representations**:
                - Physical analogies or models
                - Diagrams or visual representations
                - Hands-on activities or demonstrations

                **Assessment Applications**:
                - How examples could appear in tests
                - Problem-solving scenarios
                - Creative application challenges

                Academic Material: {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": examples_prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                results['examples'] = response.choices[0].message.content
                time.sleep(0.5)

            # Difficulty assessment (for study materials)
            if analysis_options.get('difficulty', False):
                difficulty_prompt = f"""
                As an educational psychologist and curriculum specialist, assess the difficulty level of this academic material and provide learning guidance.

                📊 DIFFICULTY LEVEL ASSESSMENT:

                **Overall Difficulty Rating**: [Scale: 1-5]
                - 1: Introductory/Basic
                - 2: Beginner with some complexity
                - 3: Intermediate
                - 4: Advanced
                - 5: Expert/Graduate level

                **Cognitive Load Analysis**:
                - **Intrinsic Load**: Complexity of core concepts
                - **Extraneous Load**: Unnecessary complexity or poor presentation
                - **Germane Load**: Effort required for meaningful learning

                **Difficulty Factors**:
                - **Vocabulary Complexity**: Technical terms and jargon level
                - **Conceptual Abstraction**: How abstract vs concrete the ideas are
                - **Prior Knowledge Requirements**: What students need to know first
                - **Logical Complexity**: Multi-step reasoning required
                - **Mathematical/Quantitative Demands**: Calculation or formula complexity

                **Learning Prerequisites**:
                - Essential background knowledge
                - Required skills and competencies
                - Recommended preparation time
                - Suggested prerequisite courses or readings

                **Time Investment Estimates**:
                - Reading and initial comprehension time
                - Deep learning and mastery time
                - Practice and application time
                - Review and retention time

                **Study Strategies by Difficulty**:
                - **Easy Sections**: Quick review techniques
                - **Moderate Sections**: Active reading and note-taking
                - **Difficult Sections**: Intensive study methods
                - **Very Difficult**: Expert guidance or tutoring needs

                **Common Learning Obstacles**:
                - Where students typically struggle
                - Misconceptions that arise
                - Breakthrough moments and insights
                - Support resources needed

                **Differentiation Suggestions**:
                - Modifications for different learning levels
                - Extension activities for advanced learners
                - Support strategies for struggling students
                - Alternative presentation methods

                Academic Material: {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": difficulty_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['difficulty'] = response.choices[0].message.content
                time.sleep(0.5)

            # Structure analysis (for assignments, essays, reports)
            if analysis_options.get('structure', False):
                structure_prompt = f"""
                As a writing and rhetoric expert, analyze the structural elements of this document to understand its organization and effectiveness.

                🏗️ STRUCTURE ANALYSIS:

                **Document Architecture**:
                - **Overall Organization**: How is the document structured?
                - **Hierarchical Elements**: Main sections, subsections, sub-points
                - **Logical Flow**: How ideas progress and connect
                - **Structural Patterns**: Problem-solution, cause-effect, chronological, etc.

                **Introduction Analysis**:
                - Hook and engagement strategies
                - Thesis or main argument presentation
                - Preview of main points
                - Context and background provision

                **Body Structure**:
                - **Paragraph Organization**: Topic sentences, development, transitions
                - **Argument Sequencing**: How arguments build upon each other
                - **Evidence Placement**: Integration of support materials
                - **Coherence Mechanisms**: Linking words, phrases, concepts

                **Conclusion Effectiveness**:
                - Summary techniques used
                - Implications and significance
                - Call to action or future directions
                - Closure and finality

                **Transitions and Connections**:
                - Between major sections
                - Between paragraphs
                - Between ideas within paragraphs
                - Forward and backward references

                **Structural Strengths**:
                - What works well organizationally
                - Clear and logical progressions
                - Effective structural choices
                - Reader-friendly elements

                **Structural Weaknesses**:
                - Organizational problems
                - Unclear connections
                - Missing elements
                - Confusing arrangements

                **Improvement Recommendations**:
                - Restructuring suggestions
                - Better transition strategies
                - Enhanced coherence methods
                - Alternative organizational approaches

                Document: {paper_text[:10000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": structure_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['structure'] = response.choices[0].message.content
                time.sleep(0.5)

            # Arguments analysis (for essays, assignments)
            if analysis_options.get('arguments', False):
                arguments_prompt = f"""
                As a critical thinking and argumentation expert, analyze the arguments presented in this document, evaluating their logic, evidence, and persuasiveness.

                💭 ARGUMENTS ANALYSIS:

                **Main Arguments Identification**:
                - **Primary Thesis/Claim**: Central argument of the document
                - **Supporting Arguments**: Key points that support the main claim
                - **Counter-Arguments**: Alternative viewpoints considered
                - **Argument Hierarchy**: How arguments relate and build upon each other

                **Argument Structure Analysis**:
                - **Premises**: Basic assumptions and starting points
                - **Evidence Types**: Data, examples, expert testimony, logical reasoning
                - **Inference Patterns**: How conclusions follow from premises
                - **Warrants**: Unstated assumptions connecting evidence to claims

                **Logical Evaluation**:
                - **Validity**: Do conclusions follow logically from premises?
                - **Soundness**: Are the premises actually true?
                - **Fallacies**: Any logical errors or weak reasoning?
                - **Consistency**: Are arguments internally consistent?

                **Evidence Assessment**:
                - **Quality of Sources**: Credibility and reliability
                - **Relevance**: How well evidence supports claims
                - **Sufficiency**: Is there enough evidence?
                - **Currency**: How recent and up-to-date is evidence?

                **Rhetorical Strategies**:
                - **Ethos**: Appeals to credibility and authority
                - **Pathos**: Emotional appeals and engagement
                - **Logos**: Logical reasoning and rational appeals
                - **Style and Tone**: How presentation affects persuasiveness

                **Argument Strengths**:
                - Most compelling points
                - Well-supported claims
                - Effective reasoning strategies
                - Persuasive elements

                **Argument Weaknesses**:
                - Weak or missing evidence
                - Logical gaps or fallacies
                - Unaddressed counter-arguments
                - Inconsistencies or contradictions

                **Overall Assessment**:
                - Persuasiveness rating (1-5 scale)
                - Target audience appropriateness
                - Potential objections and responses
                - Suggestions for strengthening arguments

                Document: {paper_text[:10000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": arguments_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['arguments'] = response.choices[0].message.content
                time.sleep(0.5)

            # Improvement suggestions (for assignments, essays)
            if analysis_options.get('improvements', False):
                improvements_prompt = f"""
                As an expert writing coach and academic mentor, provide specific, actionable suggestions for improving this document's quality, clarity, and impact.

                ✨ IMPROVEMENT SUGGESTIONS:

                **Content Improvements**:
                - **Argument Development**: Strengthen weak or underdeveloped points
                - **Evidence Integration**: Better use of sources and support materials
                - **Depth and Analysis**: Areas needing more thorough exploration
                - **Clarity and Precision**: Unclear or ambiguous sections to clarify

                **Structural Enhancements**:
                - **Organization**: Better sequencing and arrangement of ideas
                - **Transitions**: Improved connections between sections and paragraphs
                - **Introduction**: Stronger opening and thesis presentation
                - **Conclusion**: More effective closing and synthesis

                **Writing Quality**:
                - **Style and Voice**: Consistency and appropriateness improvements
                - **Word Choice**: More precise or impactful vocabulary
                - **Sentence Structure**: Variety and clarity enhancements
                - **Flow and Readability**: Smoother reading experience

                **Technical Corrections**:
                - **Grammar and Mechanics**: Any language errors to address
                - **Citation Format**: Proper source attribution and formatting
                - **Formatting Consistency**: Visual presentation improvements
                - **Length and Scope**: Appropriate depth and coverage

                **Audience Considerations**:
                - **Tone Appropriateness**: Matching audience expectations
                - **Accessibility**: Making content more understandable
                - **Engagement**: Increasing reader interest and involvement
                - **Purpose Alignment**: Better serving document objectives

                **Priority Recommendations** (Top 5):
                1. **Most Critical**: Changes with highest impact
                2. **Quick Wins**: Easy improvements with good results
                3. **Structural**: Major organizational changes needed
                4. **Content**: Substantive additions or revisions
                5. **Polish**: Final refinements for excellence

                **Revision Strategy**:
                - **First Draft Focus**: Major content and structure changes
                - **Second Draft**: Paragraph-level improvements
                - **Final Draft**: Sentence-level polishing and proofreading
                - **Timeline**: Suggested revision schedule

                **Resources and Support**:
                - **Reference Materials**: Additional sources to consult
                - **Writing Tools**: Helpful applications or techniques
                - **Feedback Sources**: Who else to consult
                - **Learning Opportunities**: Skills to develop further

                Document: {paper_text[:10000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": improvements_prompt}],
                    max_tokens=2500,
                    temperature=0.3
                )
                results['improvements'] = response.choices[0].message.content
                time.sleep(0.5)

            # Findings analysis (for reports, theses)
            if analysis_options.get('findings', False):
                findings_prompt = f"""
                As a research analyst and data interpretation expert, identify and analyze the key findings presented in this document.

                📈 KEY FINDINGS ANALYSIS:

                **Primary Findings**:
                - **Major Discoveries**: Most significant results or conclusions
                - **Statistical Results**: Quantitative findings and their significance
                - **Qualitative Insights**: Themes, patterns, and interpretations
                - **Unexpected Outcomes**: Surprising or counterintuitive results

                **Supporting Evidence**:
                - **Data Sources**: Where findings originate
                - **Methodology Impact**: How methods influenced results
                - **Sample Characteristics**: Relevance of study population
                - **Measurement Quality**: Reliability and validity considerations

                **Findings Interpretation**:
                - **Practical Significance**: Real-world importance
                - **Statistical vs. Practical**: Distinction and implications
                - **Context and Meaning**: What results actually indicate
                - **Limitations**: Constraints on interpretation

                **Comparison and Contrast**:
                - **Previous Research**: How findings relate to existing knowledge
                - **Contradictions**: Results that challenge current understanding
                - **Confirmations**: Support for established theories
                - **Novel Contributions**: New insights or perspectives

                **Implications Assessment**:
                - **Theoretical**: Impact on academic understanding
                - **Practical**: Applications for practice or policy
                - **Methodological**: Insights for future research approaches
                - **Societal**: Broader implications for society

                **Reliability and Validity**:
                - **Internal Validity**: Confidence in causal claims
                - **External Validity**: Generalizability of findings
                - **Construct Validity**: Measure appropriateness
                - **Statistical Conclusion Validity**: Appropriate statistical inference

                **Future Research Needs**:
                - **Replication**: Need for confirmation studies
                - **Extension**: Areas for further investigation
                - **Methodology**: Improvements for future studies
                - **Applications**: Implementation and testing needs

                **Communication Quality**:
                - **Clarity**: How well findings are presented
                - **Accessibility**: Understandability for different audiences
                - **Visual Aids**: Effectiveness of tables, figures, charts
                - **Summary**: Quality of conclusions and abstracts

                Document: {paper_text[:10000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": findings_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['findings'] = response.choices[0].message.content
                time.sleep(0.5)

            # Recommendations analysis (for reports, theses)
            if analysis_options.get('recommendations', False):
                recommendations_prompt = f"""
                As a strategic consultant and policy advisor, analyze and evaluate the recommendations presented in this document, assessing their feasibility and potential impact.

                💡 RECOMMENDATIONS ANALYSIS:

                **Recommendation Categories**:
                - **Policy Recommendations**: Suggested changes to rules, laws, or procedures
                - **Practice Recommendations**: Changes to current methods or approaches
                - **Research Recommendations**: Future studies or investigations needed
                - **Implementation Recommendations**: How to put changes into action

                **Recommendation Quality Assessment**:
                - **Specificity**: How clear and detailed are the suggestions?
                - **Actionability**: Can recommendations realistically be implemented?
                - **Evidence-Based**: Are suggestions supported by findings?
                - **Prioritization**: Are recommendations ranked by importance?

                **Feasibility Analysis**:
                - **Resource Requirements**: What would implementation cost?
                - **Timeline Considerations**: How long would changes take?
                - **Stakeholder Buy-in**: Who needs to support these changes?
                - **Barrier Assessment**: What obstacles might prevent success?

                **Implementation Framework**:
                - **Short-term Actions** (0-6 months): Immediate steps possible
                - **Medium-term Goals** (6-24 months): Intermediate milestones
                - **Long-term Vision** (2+ years): Ultimate objectives
                - **Success Metrics**: How to measure progress and impact

                **Stakeholder Impact**:
                - **Primary Beneficiaries**: Who would benefit most?
                - **Implementation Agents**: Who would carry out changes?
                - **Potential Resisters**: Who might oppose recommendations?
                - **Resource Providers**: Who would fund or support changes?

                **Risk Assessment**:
                - **Implementation Risks**: What could go wrong?
                - **Unintended Consequences**: Possible negative effects
                - **Mitigation Strategies**: How to reduce risks
                - **Contingency Plans**: Alternative approaches if needed

                **Effectiveness Evaluation**:
                - **Logic Model**: How recommendations lead to desired outcomes
                - **Evidence Strength**: Quality of supporting research
                - **Best Practice Alignment**: Consistency with proven approaches
                - **Innovation Assessment**: Novel or creative elements

                **Communication and Advocacy**:
                - **Key Messages**: How to present recommendations effectively
                - **Target Audiences**: Who needs to hear these suggestions?
                - **Persuasion Strategies**: How to build support
                - **Change Management**: Helping people adapt to recommendations

                **Overall Assessment**:
                - **Recommendation Strength**: Overall quality rating (1-5)
                - **Implementation Likelihood**: Probability of adoption
                - **Potential Impact**: Expected benefits if implemented
                - **Priority Ranking**: Most important recommendations to pursue

                Document: {paper_text[:10000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": recommendations_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['recommendations'] = response.choices[0].message.content
                time.sleep(0.5)

            # Main points analysis (for general academic material)
            if analysis_options.get('main_points', False):
                main_points_prompt = f"""
                As an expert summarizer and content analyst, identify and organize the main points of this academic material for clear understanding and retention.

                🎯 MAIN POINTS ANALYSIS:

                **Primary Points** (3-5 most important):
                - **Point 1**: [Title] - Detailed explanation and significance
                - **Point 2**: [Title] - Detailed explanation and significance
                - **Point 3**: [Title] - Detailed explanation and significance
                - [Additional points as needed]

                **Supporting Points** (8-12 secondary ideas):
                - How these support or elaborate on primary points
                - Evidence, examples, or explanations provided
                - Connections between supporting and primary points

                **Point Hierarchy**:
                - **Essential**: Must-know information for basic understanding
                - **Important**: Should-know information for good comprehension
                - **Supplementary**: Nice-to-know information for complete picture

                **Logical Relationships**:
                - **Causal**: Which points lead to or cause others?
                - **Sequential**: Are points presented in a specific order?
                - **Comparative**: How do points relate or contrast?
                - **Hierarchical**: Which points are more fundamental?

                **Key Takeaways**:
                - **Central Message**: The overarching theme or conclusion
                - **Practical Applications**: How points apply in real situations
                - **Learning Objectives**: What readers should accomplish
                - **Action Items**: What readers should do with this information

                **Memory and Organization Aids**:
                - **Acronyms or Mnemonics**: Memory devices for key points
                - **Visual Organization**: How points could be mapped or diagrammed
                - **Categorization**: Natural groupings of related points
                - **Progressive Disclosure**: Best order for learning points

                **Context and Background**:
                - **Assumptions**: What readers are expected to know
                - **Prerequisites**: Foundation knowledge needed
                - **Scope**: What the material covers and doesn't cover
                - **Purpose**: Why these points matter

                **Comprehension Check**:
                - **Self-Assessment Questions**: To test understanding of main points
                - **Application Exercises**: Ways to use or apply the points
                - **Discussion Prompts**: Questions for deeper engagement
                - **Connection Activities**: Linking points to prior knowledge

                Academic Material: {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": main_points_prompt}],
                    max_tokens=2000,
                    temperature=0.2
                )
                results['main_points'] = response.choices[0].message.content
                time.sleep(0.5)

            # Context analysis (for general academic material)
            if analysis_options.get('context', False):
                context_prompt = f"""
                As a contextual analyst and interdisciplinary scholar, analyze the broader context surrounding this academic material to enhance understanding and appreciation.

                🌍 CONTEXT ANALYSIS:

                **Historical Context**:
                - **Time Period**: When was this written/published?
                - **Historical Events**: Relevant contemporary events or trends
                - **Intellectual History**: Evolution of ideas and theories
                - **Precedents**: Earlier works or events that influenced this

                **Disciplinary Context**:
                - **Field of Study**: Primary academic discipline(s)
                - **Theoretical Frameworks**: Dominant theories or paradigms
                - **Methodological Traditions**: Common research approaches
                - **Key Scholars**: Important figures and their contributions

                **Social and Cultural Context**:
                - **Cultural Values**: Prevailing beliefs and attitudes
                - **Social Issues**: Relevant societal concerns or problems
                - **Political Climate**: Government policies or political trends
                - **Economic Factors**: Financial or economic influences

                **Institutional Context**:
                - **Author Background**: Institution, credentials, perspectives
                - **Publication Context**: Journal, publisher, intended audience
                - **Funding Sources**: Who sponsored or supported the work
                - **Academic Networks**: Professional relationships and influences

                **Intellectual Context**:
                - **Ongoing Debates**: Current controversies or discussions
                - **Competing Theories**: Alternative explanations or approaches
                - **Knowledge Gaps**: What was unknown or disputed
                - **Paradigm Status**: Revolutionary, normal science, or crisis?

                **Contemporary Relevance**:
                - **Current Applications**: How ideas apply today
                - **Modern Developments**: Recent advances or changes
                - **Ongoing Influence**: Impact on current thinking
                - **Future Directions**: Where ideas might lead

                **Comparative Context**:
                - **Similar Works**: Related materials or studies
                - **International Perspectives**: How other cultures view these ideas
                - **Cross-Disciplinary Connections**: Links to other fields
                - **Scale Considerations**: Individual, group, organizational, societal levels

                **Critical Context**:
                - **Assumptions and Biases**: Unstated beliefs or limitations
                - **Power Dynamics**: Who benefits from these ideas?
                - **Excluded Voices**: Whose perspectives are missing?
                - **Ethical Considerations**: Moral or ethical implications

                **Learning Context**:
                - **Prerequisites**: What background knowledge helps?
                - **Learning Objectives**: Why students encounter this material
                - **Skill Development**: What abilities this material builds
                - **Assessment Context**: How understanding is typically evaluated

                Academic Material: {paper_text[:8000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": context_prompt}],
                    max_tokens=2000,
                    temperature=0.3
                )
                results['context'] = response.choices[0].message.content
                time.sleep(0.5)

            # Always provide comprehensive analysis if detailed is requested
            if analysis_options.get('detailed', True):
                detailed_prompt = f"""
                As a senior academic researcher with expertise in comprehensive literature analysis, provide an in-depth analysis of this research paper. Your analysis should demonstrate the sophistication expected of a leading expert in the field.

                📋 COMPREHENSIVE RESEARCH ANALYSIS:

                **Research Context & Significance**:
                - How does this research fit within the broader landscape of the field?
                - What makes this contribution novel or important?
                - How does it build upon or challenge existing knowledge?

                **Theoretical Framework**:
                - What theoretical foundations underpin this research?
                - How well are these theories applied and integrated?
                - Are there theoretical gaps or opportunities for improvement?

                **Methodological Excellence & Concerns**:
                - Detailed evaluation of research design appropriateness
                - Assessment of data quality and analytical rigor
                - Identification of potential biases or confounding factors
                - Evaluation of statistical power and effect sizes

                **Results Interpretation**:
                - Critical analysis of findings and their significance
                - Assessment of how well results support the conclusions
                - Identification of alternative interpretations
                - Evaluation of practical vs statistical significance

                **Strengths & Innovations**:
                - What does this research do particularly well?
                - What novel contributions does it make?
                - How does it advance methodological or theoretical understanding?

                **Limitations & Concerns**:
                - Critical assessment of study limitations
                - Potential threats to validity
                - Generalizability concerns
                - Methodological or analytical weaknesses

                **Broader Implications**:
                - How do these findings impact theory and practice?
                - What are the policy or practical applications?
                - How might this influence future research directions?

                **Quality Assessment**:
                - Overall research quality rating (1-10 scale) with detailed justification
                - Assessment of publication worthiness and potential impact
                - Recommendations for improvement

                **Future Research Agenda**:
                - Specific next steps that would build on this work
                - Important questions that remain unanswered
                - Methodological innovations that could advance the field

                Provide analysis at the level expected for a top-tier academic journal review. Be thorough, critical, and constructive.

                RESEARCH PAPER TEXT:
                {paper_text[:12000]}
                """
                
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": detailed_prompt}],
                    max_tokens=3000,
                    temperature=0.1
                )
                results['detailed_analysis'] = response.choices[0].message.content
                time.sleep(0.5)
            
            return results
            
        except Exception as e:
            logger.error(f"Error during paper analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def suggest_related_papers(self, analyzed_content: str) -> str:
        """
        Suggest related papers based on the analyzed content.
        
        Args:
            analyzed_content: The content that was analyzed
            
        Returns:
            String containing suggested related papers
        """
        try:
            prompt = f"""
            Based on this research analysis, suggest 8-10 highly relevant papers that researchers should read to gain deeper understanding of this topic. 

            For each suggested paper:
            - Provide likely author names and publication year
            - Give a realistic title that would exist in this research area
            - Explain why this paper would be relevant (2-3 sentences)
            - Indicate what specific aspects it would illuminate

            Focus on:
            1. Foundational/seminal works in this area
            2. Recent advances and methodology papers
            3. Review papers that provide broader context
            4. Papers with contrasting viewpoints or approaches
            5. Papers that extend or apply these findings

            Format as a numbered list with clear structure.

            Analyzed Content:
            {analyzed_content[:8000]}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error suggesting related papers: {e}")
            return f"Error generating related papers suggestions: {str(e)}"

    def generate_research_questions(self, analyzed_content: str) -> str:
        """
        Generate research questions based on the analyzed content.
        
        Args:
            analyzed_content: The content that was analyzed
            
        Returns:
            String containing generated research questions
        """
        try:
            prompt = f"""
            Based on this research analysis, generate 10-12 thought-provoking research questions that could guide future research in this area.

            Create questions that:
            1. **Build on current findings** - Extend or test the results in new contexts
            2. **Address identified limitations** - Target specific methodological or conceptual gaps
            3. **Explore mechanisms** - Dig deeper into how or why phenomena occur
            4. **Cross-disciplinary connections** - Link to other fields or perspectives
            5. **Practical applications** - Connect theory to real-world implementation
            6. **Scale considerations** - Address different levels of analysis
            7. **Temporal dynamics** - Explore changes over time

            For each question:
            - Make it specific and actionable
            - Ensure it can be reasonably investigated
            - Explain briefly why it's important (1-2 sentences)
            - Suggest the most appropriate methodology

            Format as a numbered list with clear explanations.

            Analyzed Content:
            {analyzed_content[:8000]}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.4
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating research questions: {e}")
            return f"Error generating research questions: {str(e)}"

    def build_hypotheses(self, analyzed_content: str) -> str:
        """Generate research hypotheses based on analyzed content."""
        try:
            prompt = f"""
            As a research methodology expert, generate 6-8 testable research hypotheses based on this analyzed content.

            🧠 RESEARCH HYPOTHESES GENERATION:

            **Hypothesis Development Framework**:
            For each hypothesis, provide:
            1. **Clear Statement**: Specific, testable prediction
            2. **Theoretical Basis**: Why this hypothesis makes sense
            3. **Variables**: Independent and dependent variables
            4. **Methodology**: How it could be tested
            5. **Expected Outcomes**: Predicted results and implications

            **Categories of Hypotheses**:
            1. **Causal Hypotheses** (2-3): Test cause-effect relationships
            2. **Correlational Hypotheses** (2-3): Test relationships between variables
            3. **Comparative Hypotheses** (1-2): Compare groups or conditions
            4. **Descriptive Hypotheses** (1-2): Describe phenomena or patterns

            **Quality Criteria**:
            - Testable and falsifiable
            - Based on logical reasoning from the content
            - Specific enough to guide research design
            - Significant enough to contribute new knowledge

            Analyzed Content: {analyzed_content[:8000]}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.3
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error building hypotheses: {e}")
            return f"Error building hypotheses: {str(e)}"

    def generate_research_proposal(self, analyzed_content: str) -> str:
        """Generate a research proposal based on analyzed content."""
        try:
            prompt = f"""
            As an experienced grant writer and research supervisor, draft a comprehensive research proposal based on this analyzed content.

            📋 RESEARCH PROPOSAL STRUCTURE:

            **1. Title & Abstract** (200 words)
            - Compelling title
            - Concise summary of the proposed research

            **2. Research Problem & Significance** (300 words)
            - What specific problem will be addressed?
            - Why is this research important and timely?
            - What gap in knowledge will be filled?

            **3. Literature Review Summary** (250 words)
            - Key studies and theoretical foundations
            - What has been done and what's missing
            - How this research builds on existing work

            **4. Research Questions & Hypotheses** (200 words)
            - 2-3 specific research questions
            - Clear, testable hypotheses
            - Expected outcomes

            **5. Methodology** (400 words)
            - Research design and approach
            - Participants/sample
            - Data collection methods
            - Analysis plan
            - Timeline

            **6. Expected Contributions** (150 words)
            - Theoretical contributions
            - Practical applications
            - Policy implications

            **7. Budget Considerations** (100 words)
            - Major cost categories
            - Justification for resources

            Make it compelling and feasible. Use professional academic language.

            Analyzed Content: {analyzed_content[:10000]}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.2
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error generating research proposal: {e}")
            return f"Error generating research proposal: {str(e)}"

    def generate_flashcards(self, content: str) -> str:
        """Generate flashcards for study purposes."""
        try:
            prompt = f"""
            Create 15-20 high-quality flashcards based on this academic content for effective studying.

            🃏 FLASHCARD CREATION:

            **Format for each flashcard**:
            **Card #**: [Front] | [Back]

            **Types of flashcards to create**:
            1. **Definition Cards** (5-7 cards): Key terms and their definitions
            2. **Concept Cards** (4-5 cards): Important concepts and explanations
            3. **Process Cards** (3-4 cards): Steps, procedures, or methodologies
            4. **Comparison Cards** (2-3 cards): Contrasting ideas or approaches
            5. **Application Cards** (2-3 cards): Examples and real-world applications

            **Quality guidelines**:
            - Front: Clear, specific question or term
            - Back: Complete but concise answer
            - Avoid yes/no questions
            - Make each card test one concept
            - Use active recall principles

            Content: {content[:8000]}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.2
            )
            return response.choices[0].message.content
            
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
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.2
            )
            return response.choices[0].message.content
            
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
            - Review checklist

            **VII. Additional Resources**
            - Suggested supplementary materials
            - Related topics to explore

            Make it comprehensive but organized for efficient studying.

            Content: {content[:10000]}
            """
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,
                temperature=0.2
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error building study guide: {e}")
            return f"Error building study guide: {str(e)}"

    def analyze_class_material(self, content: str, material_type: str = "general") -> str:
        """Analyze class material for study purposes."""
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
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2500,
                temperature=0.2
            )
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error analyzing class material: {e}")
            return f"Error analyzing class material: {str(e)}"
            