# 🎓 Academic AI Assistant

[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF6B6B)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-4285F4)](https://ai.google.dev/)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB)](https://python.org/)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Transform research papers AND class materials into actionable insights with cutting-edge AI analysis. Perfect for researchers, students, and academics who need comprehensive document analysis and study tools.

![Academic AI Assistant Demo](https://8pkvharc8cebbrhcpjhazn.streamlit.app/)

## ✨ Dual-Purpose Features

### 🔬 **Research Analysis Tools**
Transform dense research papers into comprehensive insights:

- **📄 Smart PDF Processing** - Advanced text extraction with structure preservation
- **📝 Executive Summaries** - Concise, accessible overviews of complex research  
- **🔬 Methodology Breakdown** - Deep-dive analysis of research methods and procedures
- **📚 Citation Network Analysis** - Automatic reference extraction and relationship mapping
- **🔍 Research Gap Identification** - AI-powered discovery of unexplored opportunities
- **🏷️ Keyword Extraction** - Key terms, concepts, and technical vocabulary
- **� Related Paper Suggestions** - AI-powered discovery of similar research
- **❓ Research Question Generator** - Generate meaningful questions for future investigation
- **💡 Hypothesis Builder** - Develop testable hypotheses from research findings
- **📋 Research Proposal Assistant** - AI-guided proposal drafting and grant applications

### 📚 **Student Study Tools** 
Turn class materials into effective study resources:

- **📇 Smart Flashcards** - Extract key definitions and concepts for active recall
- **❓ Practice Questions** - Generate multiple choice, short answer, and essay questions
- **📖 Study Guides** - Create comprehensive guides with organized topics and summaries
- **📊 Material Analysis** - Assess difficulty levels and provide study recommendations
- **⏰ Exam Focus Predictions** - Identify likely test areas and important concepts
- **💾 Export Everything** - Download all study materials as formatted files

## 🎯 **Perfect For**

### 🎓 **Researchers & Academics**
- Quickly understand complex papers in your field
- Generate research ideas and hypotheses
- Draft compelling research proposals
- Identify gaps in current literature

### 📚 **Students** 
- Break down difficult textbooks and course materials
- Create effective study materials (flashcards, guides, questions)
- Prepare for exams with AI-generated practice questions
- Understand complex academic papers for assignments

### 🔬 **Research Teams**
- Streamline literature review processes
- Collaborative analysis and discussion
- Stay updated with latest developments
- Generate new research directions

### � **Educators**
- Quickly analyze teaching materials
- Create study resources for students
- Develop comprehensive course content
- Assess material difficulty and learning objectives

## 🚀 Quick Start

### Prerequisites
- Python 3.11 or higher
- Groq API key ([Get one here](https://console.groq.com/keys))

### Installation

```bash
# Clone the repository
git clone https://github.com/AndrewPuig77/AI-Academic-Assistant-groq-.git
cd AI-Academic-Assistant-groq-

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your API key
cp .env.example .env
# Edit .env and add your Groq API key
```

### Configuration

1. **Get Your Groq API Key**
   - Visit [Groq Console](https://console.groq.com/keys)
   - Create a new API key (FREE tier available!)
   - Copy the key

2. **Configure the Application**
   ```bash
   # Edit the .env file
   GROQ_API_KEY=your_actual_api_key_here
   ```

### Run the Application

```bash
# Start the Streamlit app
streamlit run main.py
```

Open your browser to **http://localhost:8501** and start analyzing! 🎉

## 📖 Usage

### For Research Papers
1. **📤 Upload** - Drop a research paper PDF
2. **⚙️ Configure** - Select analysis options (summary, methodology, citations, etc.)
3. **🔬 Research Tools** - Generate questions, hypotheses, and research proposals
4. **📥 Export** - Download comprehensive analysis reports

### For Class Materials  
1. **� Upload** - Drop textbook chapters, lecture notes, or course handouts
2. **📚 Study Tools** - Choose flashcards, practice questions, or study guides
3. **� Analyze** - Get difficulty assessments and study recommendations
4. **� Export** - Download complete study packages

## 🛠️ Technology Stack

### Core Technologies
- **🐍 Python 3.11+** - Modern Python with advanced features
- **🎨 Streamlit** - Beautiful, interactive web interface with modern black theme
-
- **📄 PyMuPDF** - Advanced PDF processing and text extraction
- **� Plotly, Matplotlib, Seaborn** - Interactive visualizations and charts
- **☁️ WordCloud** - Visual text analysis and keyword representation
- **� ReportLab** - PDF export functionality

##  License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">
  <p><strong>Built with ❤️ for researchers AND students</strong></p>
  <p><em>Making academic success accessible to everyone with lightning-fast AI</em></p>
  
  ![Made with Love](https://img.shields.io/badge/Made%20with-❤️-red.svg)
  ![Open Source](https://img.shields.io/badge/Open%20Source-💚-brightgreen.svg)
  ![Student Friendly](https://img.shields.io/badge/Student-Friendly-blue.svg)
  ![Lightning Fast](https://img.shields.io/badge/Lightning-Fast-yellow.svg)
</div>
