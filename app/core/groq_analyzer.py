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
            # Generate sophisticated summary if requested
            if analysis_options.get('summary', False):
                summary_prompt = f"""
                As an expert research analyst with deep expertise in academic literature, provide a comprehensive but concise summary of this research paper. Your analysis should demonstrate sophisticated understanding of the research domain.

                📋 RESEARCH SUMMARY:

                • **Main Research Question/Problem**: What specific problem or gap in knowledge does this paper address? What makes this research question significant in the broader field?

                • **Key Methodology**: Describe the research approach with attention to methodological rigor. What specific methods were employed and why were they appropriate for addressing the research question?

                • **Major Findings**: What are the most significant and novel results? How do these findings advance our understanding beyond previous work?

                • **Practical Implications**: How can these findings be applied in practice? What are the real-world implications for practitioners, policymakers, or future research?

                • **Limitations**: What are the key limitations explicitly mentioned or implicitly evident? How do these limitations affect the interpretation and generalizability of results?

                Write with the sophistication expected of a senior researcher. Use precise academic language while remaining accessible. Demonstrate critical thinking and contextual understanding.
                Limit to 400-500 words.

                RESEARCH PAPER TEXT:
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
                methodology_prompt = f"""
                As a methodological expert and research design specialist, provide a comprehensive analysis of this study's methodology. Your analysis should demonstrate deep understanding of research design principles and methodological rigor.

                🔬 METHODOLOGY ANALYSIS:

                **Research Design**: What type of study/experiment is this? Evaluate the appropriateness of the design for addressing the research question. Consider whether alternative designs might have been more suitable.

                **Data Collection**: Analyze the data collection methods in detail. What instruments, procedures, or techniques were used? Evaluate their validity and reliability. Were there any potential sources of bias in data collection?

                **Sample/Participants**: Examine the sample characteristics, size, and selection methods. Is the sample representative of the target population? What are the implications of the sampling approach for generalizability?

                **Variables and Measurements**: Identify and analyze the independent, dependent, and control variables. How were key constructs operationalized and measured? Evaluate the quality of these measurements.

                **Controls and Confounds**: What controls were implemented to address potential confounding variables? Were there uncontrolled factors that could have influenced the results?

                **Statistical Analysis**: Evaluate the statistical methods used. Were they appropriate for the data type and research design? Were assumptions met? Were effect sizes reported?

                **Validity Assessment**: 
                - **Internal Validity**: To what extent can we be confident that the observed effects are due to the manipulated variables rather than other factors?
                - **External Validity**: How generalizable are these findings to other populations, settings, or conditions?
                - **Construct Validity**: How well do the measures capture the intended theoretical constructs?

                **Reproducibility**: Could this study be replicated by other researchers? What information is provided to enable replication?

                **Methodology Strength Rating**: Provide an overall rating (1-5 scale) of the methodological rigor with detailed justification.

                **Suggested Improvements**: What specific methodological improvements could strengthen future research in this area?

                Demonstrate expertise in research methodology. Be critical but fair, acknowledging both strengths and limitations.

                RESEARCH PAPER TEXT:
                {paper_text[:10000]}
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
                As a research librarian and subject matter expert, extract and categorize key terms and concepts from this research paper. Your analysis should facilitate literature searches and demonstrate deep understanding of the research domain.

                🏷️ RESEARCH KEYWORDS:

                **Primary Keywords** (5-7 terms):
                The most important terms that capture the core focus of this research. These should be terms that researchers in this field would immediately recognize as central.

                **Technical Terminology** (8-12 terms):
                Specialized terms, methodological concepts, and domain-specific vocabulary that are essential for understanding this research.

                **Theoretical Concepts** (5-8 terms):
                Underlying theories, frameworks, and conceptual approaches that inform this research.

                **Methodological Terms** (5-8 terms):
                Research methods, analytical techniques, and methodological approaches used in this study.

                **Field-Specific Language** (5-10 terms):
                Discipline-specific terminology that situates this research within its broader academic field.

                **Related Research Areas** (5-7 areas):
                Adjacent fields or research domains that connect to this work.

                **Search Strategy Recommendations**:
                - Suggest optimal database search terms for finding related literature
                - Provide Boolean search combinations that would be most effective
                - Recommend MeSH terms or subject headings where applicable

                **Conceptual Hierarchy**:
                Organize the key concepts into a hierarchical structure showing relationships between broader and more specific terms.

                **Alternative Terminology**:
                List alternative terms or synonyms that might be used in the literature to refer to the same concepts.

                RESEARCH PAPER TEXT:
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
            