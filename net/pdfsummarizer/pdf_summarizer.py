#!/usr/bin/env python3
"""
PDF Directory Summarizer with OCR Support
Processes all PDF files in a directory and creates detailed summaries
Includes OCR capabilities for scanned documents
"""

import os
import sys
import argparse
from datetime import datetime
import PyPDF2
import pdfplumber
from pathlib import Path
import re
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings('ignore')

# OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("Warning: OCR libraries not installed. Install with: pip install pytesseract pdf2image pillow")
    print("Also ensure Tesseract OCR is installed on your system")

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class PDFSummarizer:
    def __init__(self, output_file='pdf_summaries.txt', use_ocr=True, ocr_language='eng'):
        self.output_file = output_file
        self.stop_words = set(stopwords.words('english'))
        self.use_ocr = use_ocr and OCR_AVAILABLE
        self.ocr_language = ocr_language
        
    def extract_text_with_ocr(self, pdf_path):
        """Extract text from PDF using OCR on each page"""
        text = ""
        try:
            print(f"  Performing OCR on {os.path.basename(pdf_path)}...")
            
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            
            # Process each page
            for i, image in enumerate(images):
                print(f"  Processing page {i+1}/{len(images)} with OCR...")
                
                # Perform OCR on the page
                page_text = pytesseract.image_to_string(
                    image, 
                    lang=self.ocr_language,
                    config='--psm 3'  # Page segmentation mode for automatic page segmentation
                )
                
                # Add page marker
                text += f"\n[Page {i+1}]\n{page_text}\n"
                
                # Also try to extract any additional data
                try:
                    # Get detailed OCR data
                    ocr_data = pytesseract.image_to_data(
                        image, 
                        lang=self.ocr_language,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Extract confidence scores
                    confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        if avg_confidence < 50:
                            print(f"  Warning: Low OCR confidence on page {i+1} ({avg_confidence:.1f}%)")
                except:
                    pass
                    
        except Exception as e:
            print(f"  OCR error: {str(e)}")
            return ""
        
        return text
    
    def extract_text_pypdf2(self, pdf_path):
        """Extract text using PyPDF2"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    text += f"\n[Page {page_num + 1}]\n{page_text}\n"
                    
        except Exception as e:
            print(f"  PyPDF2 error: {str(e)}")
        return text
    
    def extract_text_pdfplumber(self, pdf_path):
        """Extract text using pdfplumber (better for complex layouts)"""
        text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n[Page {i + 1}]\n{page_text}\n"
                        
                    # Also try to extract tables
                    tables = page.extract_tables()
                    if tables:
                        text += "\n[Tables found on this page]\n"
                        for table in tables:
                            for row in table:
                                text += " | ".join(str(cell) if cell else "" for cell in row) + "\n"
                            text += "\n"
                            
        except Exception as e:
            print(f"  pdfplumber error: {str(e)}")
        return text
    
    def is_scanned_pdf(self, pdf_path, text_pypdf, text_plumber):
        """Determine if PDF is likely scanned (contains images instead of text)"""
        # If both methods extract very little text, it's likely scanned
        total_text = text_pypdf + text_plumber
        word_count = len(total_text.split())
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                # If less than 50 words per page on average, likely scanned
                if num_pages > 0 and word_count / num_pages < 50:
                    return True
        except:
            pass
            
        return False
    
    def extract_metadata(self, pdf_path):
        """Extract PDF metadata"""
        metadata = {}
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                info = pdf_reader.metadata
                if info:
                    metadata = {
                        'title': info.get('/Title', 'N/A'),
                        'author': info.get('/Author', 'N/A'),
                        'subject': info.get('/Subject', 'N/A'),
                        'creator': info.get('/Creator', 'N/A'),
                        'producer': info.get('/Producer', 'N/A'),
                        'creation_date': str(info.get('/CreationDate', 'N/A')),
                        'modification_date': str(info.get('/ModDate', 'N/A')),
                        'pages': len(pdf_reader.pages)
                    }
                else:
                    metadata['pages'] = len(pdf_reader.pages)
        except Exception as e:
            print(f"  Metadata extraction error: {str(e)}")
        return metadata
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]]', '', text)
        # Fix OCR common errors
        text = re.sub(r'\bl\b', 'I', text)  # Common OCR error: l instead of I
        text = re.sub(r'\bO\b', '0', text)  # Common OCR error: O instead of 0
        return text.strip()
    
    def extract_key_sentences(self, text, num_sentences=10):
        """Extract most important sentences based on word frequency"""
        sentences = sent_tokenize(text)
        if len(sentences) <= num_sentences:
            return sentences
        
        # Calculate word frequencies
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalnum() and w not in self.stop_words and len(w) > 2]
        word_freq = Counter(words)
        
        # Score sentences
        sentence_scores = {}
        for sentence in sentences:
            words_in_sentence = word_tokenize(sentence.lower())
            score = 0
            word_count = 0
            for word in words_in_sentence:
                if word in word_freq:
                    score += word_freq[word]
                    word_count += 1
            if word_count > 0:
                sentence_scores[sentence] = score / word_count
        
        # Get top sentences
        sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)
        return [sent[0] for sent in sorted_sentences[:num_sentences]]
    
    def analyze_structure(self, text):
        """Analyze document structure"""
        lines = text.split('\n')
        
        # Detect potential headings (lines with fewer words, possibly in caps)
        potential_headings = []
        for line in lines:
            line = line.strip()
            if line and len(line.split()) <= 10:
                if line.isupper() or line.istitle() or re.match(r'^\d+\.?\s+\w+', line):
                    potential_headings.append(line)
        
        # Detect sections
        sections = []
        current_section = ""
        for line in lines:
            if line.strip() in potential_headings:
                if current_section:
                    sections.append(current_section)
                current_section = f"\n[SECTION: {line.strip()}]\n"
            else:
                current_section += line + "\n"
        if current_section:
            sections.append(current_section)
        
        return potential_headings, sections
    
    def generate_detailed_summary(self, pdf_path):
        """Generate a comprehensive summary of the PDF"""
        print(f"\nProcessing: {pdf_path}")
        
        # Extract text using both methods
        print("  Attempting standard text extraction...")
        text_pypdf = self.extract_text_pypdf2(pdf_path)
        text_plumber = self.extract_text_pdfplumber(pdf_path)
        
        # Check if PDF is scanned
        is_scanned = self.is_scanned_pdf(pdf_path, text_pypdf, text_plumber)
        ocr_used = False
        
        # Use the longer extraction or perform OCR if needed
        if is_scanned and self.use_ocr:
            print("  PDF appears to be scanned. Using OCR...")
            text = self.extract_text_with_ocr(pdf_path)
            ocr_used = True
            if not text.strip():
                # Fall back to standard extraction
                text = text_pypdf if len(text_pypdf) > len(text_plumber) else text_plumber
        else:
            text = text_pypdf if len(text_pypdf) > len(text_plumber) else text_plumber
        
        if not text.strip():
            return f"ERROR: Could not extract text from {pdf_path} (tried standard extraction{' and OCR' if self.use_ocr else ''})\n\n"
        
        # Clean text
        text = self.clean_text(text)
        
        # Extract metadata
        metadata = self.extract_metadata(pdf_path)
        
        # Analyze structure
        headings, sections = self.analyze_structure(text)
        
        # Calculate statistics
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        word_count = len([w for w in words if w.isalnum()])
        
        # Extract key sentences
        key_sentences = self.extract_key_sentences(text, num_sentences=15)
        
        # Find most common words (excluding stopwords)
        content_words = [w.lower() for w in words if w.isalnum() and w.lower() not in self.stop_words and len(w) > 2]
        common_words = Counter(content_words).most_common(20)
        
        # Build detailed summary
        summary = f"""
{'='*80}
FILE: {os.path.basename(pdf_path)}
PATH: {pdf_path}
EXTRACTION METHOD: {'OCR' if ocr_used else 'Standard Text Extraction'}
{'='*80}

METADATA:
---------
Title: {metadata.get('title', 'N/A')}
Author: {metadata.get('author', 'N/A')}
Subject: {metadata.get('subject', 'N/A')}
Pages: {metadata.get('pages', 'N/A')}
Creation Date: {metadata.get('creation_date', 'N/A')}
Modification Date: {metadata.get('modification_date', 'N/A')}

DOCUMENT STATISTICS:
-------------------
Total Pages: {metadata.get('pages', 'N/A')}
Total Sentences: {len(sentences)}
Total Words: {word_count}
Average Words per Sentence: {word_count/len(sentences):.1f} if sentences else 0
Extraction Type: {'OCR (Scanned Document)' if ocr_used else 'Native Text'}

IDENTIFIED HEADINGS/SECTIONS:
----------------------------
"""
        for i, heading in enumerate(headings[:15], 1):
            summary += f"{i}. {heading}\n"
        
        if not headings:
            summary += "No clear headings identified\n"
        
        summary += f"""
KEY TERMS (Top 20 Most Frequent):
---------------------------------
"""
        for word, freq in common_words:
            summary += f"- {word}: {freq} occurrences\n"
        
        summary += f"""
CONTENT SUMMARY (Key Sentences):
--------------------------------
"""
        for i, sentence in enumerate(key_sentences, 1):
            summary += f"{i}. {sentence.strip()}\n\n"
        
        # Add first 500 words for context
        summary += f"""
DOCUMENT BEGINNING (First 500 words):
------------------------------------
{' '.join(text.split()[:500])}...

DOCUMENT ENDING (Last 300 words):
---------------------------------
...{' '.join(text.split()[-300:])}

"""
        
        # If sections were identified, include section summaries
        if len(sections) > 1:
            summary += "SECTION SUMMARIES:\n"
            summary += "-----------------\n"
            for i, section in enumerate(sections[:5], 1):  # First 5 sections
                section_sentences = sent_tokenize(section)[:3]  # First 3 sentences
                summary += f"\nSection {i}:\n"
                for sent in section_sentences:
                    summary += f"  {sent.strip()}\n"
        
        # Add OCR confidence note if OCR was used
        if ocr_used:
            summary += f"\nNOTE: This document was processed using OCR. Text accuracy may vary.\n"
        
        summary += "\n" + "="*80 + "\n\n"
        
        return summary
    
    def process_directory(self, directory_path):
        """Process all PDFs in a directory"""
        directory = Path(directory_path)
        
        if not directory.exists():
            print(f"Error: Directory {directory_path} does not exist")
            return
        
        pdf_files = list(directory.glob("*.pdf")) + list(directory.glob("*.PDF"))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return
        
        print(f"Found {len(pdf_files)} PDF files to process")
        print(f"OCR Support: {'Enabled' if self.use_ocr else 'Disabled'}")
        
        # Create output file
        with open(self.output_file, 'w', encoding='utf-8') as out_file:
            # Write header
            out_file.write(f"""PDF DIRECTORY SUMMARY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Directory: {directory_path}
Total Files: {len(pdf_files)}
OCR Enabled: {self.use_ocr}
OCR Language: {self.ocr_language if self.use_ocr else 'N/A'}
{'='*80}

TABLE OF CONTENTS:
""")
            
            # Write table of contents
            for i, pdf_file in enumerate(pdf_files, 1):
                out_file.write(f"{i}. {pdf_file.name}\n")
            
            out_file.write("\n" + "="*80 + "\n\n")
            
            # Process each PDF
            processed = 0
            errors = 0
            
            for i, pdf_file in enumerate(pdf_files, 1):
                print(f"\nProcessing file {i}/{len(pdf_files)}: {pdf_file.name}")
                
                try:
                    summary = self.generate_detailed_summary(pdf_file)
                    out_file.write(summary)
                    out_file.flush()  # Ensure data is written
                    processed += 1
                except Exception as e:
                    error_msg = f"\nERROR processing {pdf_file.name}: {str(e)}\n\n"
                    print(error_msg)
                    out_file.write(error_msg)
                    errors += 1
            
            # Write footer
            out_file.write(f"""
{'='*80}
END OF REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Total Files Processed: {processed}
Errors: {errors}
Output File: {self.output_file}
{'='*80}
""")
        
        print(f"\n{'='*50}")
        print(f"Summary report saved to: {self.output_file}")
        print(f"Successfully processed: {processed}/{len(pdf_files)} files")
        if errors > 0:
            print(f"Errors encountered: {errors}")

def main():
    parser = argparse.ArgumentParser(
        description='Generate detailed summaries of all PDFs in a directory with OCR support',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pdf_summarizer.py /path/to/pdfs
  python pdf_summarizer.py /path/to/pdfs -o my_summary.txt
  python pdf_summarizer.py /path/to/pdfs --no-ocr
  python pdf_summarizer.py /path/to/pdfs --ocr-lang deu  # German OCR

OCR Installation:
  pip install pytesseract pdf2image pillow
  
  Also install Tesseract OCR:
  - Ubuntu/Debian: sudo apt-get install tesseract-ocr
  - macOS: brew install tesseract
  - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  
  For additional languages:
  - Ubuntu/Debian: sudo apt-get install tesseract-ocr-[lang]
  - Example: sudo apt-get install tesseract-ocr-deu  # German
        """
    )
    
    parser.add_argument('directory', help='Path to directory containing PDF files')
    parser.add_argument('-o', '--output', default='pdf_summaries.txt', 
                       help='Output file name (default: pdf_summaries.txt)')
    parser.add_argument('--no-ocr', action='store_true',
                       help='Disable OCR processing for scanned documents')
    parser.add_argument('--ocr-lang', default='eng',
                       help='OCR language (default: eng). Use "eng+deu" for multiple languages')
    
    args = parser.parse_args()
    
    # Create summarizer and process directory
    summarizer = PDFSummarizer(
        output_file=args.output,
        use_ocr=not args.no_ocr,
        ocr_language=args.ocr_lang
    )
    summarizer.process_directory(args.directory)

if __name__ == "__main__":
    main()
