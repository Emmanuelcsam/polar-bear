#!/usr/bin/env python3
"""
Document Compressor for AI Consumption
Intelligently compresses PDFs and text files while preserving all essential information
"""

import os
import sys
import subprocess
import importlib
import json
import hashlib
import re
from datetime import datetime
from pathlib import Path

# Dependency checker and installer
def check_and_install_dependencies():
    """Check for required packages and install if missing"""
    required_packages = {
        'PyPDF2': 'PyPDF2',
        'nltk': 'nltk',
        'tiktoken': 'tiktoken',
        'chardet': 'chardet'
    }
    
    print("\n🔍 Checking dependencies...")
    missing_packages = []
    
    for package_import, package_name in required_packages.items():
        try:
            importlib.import_module(package_import)
            print(f"✅ {package_name} is installed")
        except ImportError:
            missing_packages.append(package_name)
            print(f"❌ {package_name} is missing")
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("✅ All dependencies installed successfully!\n")
    else:
        print("✅ All dependencies are already installed!\n")

# Run dependency check
check_and_install_dependencies()

# Now import the packages
import PyPDF2
import nltk
import tiktoken
import chardet
from collections import defaultdict

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📥 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

class DocumentCompressor:
    def __init__(self):
        self.config = {}
        self.stats = {
            'total_files': 0,
            'pdf_files': 0,
            'text_files': 0,
            'total_chars_before': 0,
            'total_chars_after': 0,
            'failed_files': []
        }
        
    def get_user_config(self):
        """Interactive configuration"""
        print("=== Document Compressor Configuration ===\n")
        
        # Get directory path
        while True:
            dir_path = input("📁 Enter the path to your directory containing PDFs/text files: ").strip()
            if os.path.isdir(dir_path):
                self.config['input_dir'] = dir_path
                break
            print("❌ Directory not found. Please try again.")
        
        # Output file name
        default_output = "compressed_documents.json"
        output_name = input(f"\n📄 Output filename (default: {default_output}): ").strip()
        self.config['output_file'] = output_name if output_name else default_output
        
        # Chunk size
        default_chunk = "2000"
        chunk_size = input(f"\n📏 Maximum chunk size in tokens (default: {default_chunk}): ").strip()
        try:
            self.config['chunk_size'] = int(chunk_size) if chunk_size else int(default_chunk)
        except ValueError:
            print(f"Invalid input, using default: {default_chunk}")
            self.config['chunk_size'] = int(default_chunk)
        
        # Preserve formatting
        preserve = input("\n🎨 Preserve original formatting? (y/n, default: y): ").strip().lower()
        self.config['preserve_formatting'] = preserve != 'n'
        
        # Include metadata
        metadata = input("\n📊 Include file metadata? (y/n, default: y): ").strip().lower()
        self.config['include_metadata'] = metadata != 'n'
        
        # Compression level
        print("\n🗜️  Compression level:")
        print("1. Light - Minimal compression, preserves most whitespace")
        print("2. Medium - Moderate compression, normalizes whitespace")
        print("3. Heavy - Maximum compression, removes redundancy")
        comp_level = input("Choose (1-3, default: 2): ").strip()
        try:
            self.config['compression_level'] = int(comp_level) if comp_level and 1 <= int(comp_level) <= 3 else 2
        except ValueError:
            self.config['compression_level'] = 2
        
        # Summary
        print("\n=== Configuration Summary ===")
        print(f"📁 Input Directory: {self.config['input_dir']}")
        print(f"📄 Output File: {self.config['output_file']}")
        print(f"📏 Chunk Size: {self.config['chunk_size']} tokens")
        print(f"🎨 Preserve Formatting: {'Yes' if self.config['preserve_formatting'] else 'No'}")
        print(f"📊 Include Metadata: {'Yes' if self.config['include_metadata'] else 'No'}")
        print(f"🗜️  Compression Level: {self.config['compression_level']}")
        
        confirm = input("\n✅ Proceed with these settings? (y/n): ").strip().lower()
        return confirm != 'n'
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
            
            return text, num_pages
        except Exception as e:
            print(f"⚠️  Error reading PDF {file_path}: {str(e)}")
            return None, 0
    
    def extract_text_from_file(self, file_path):
        """Extract text from text file with encoding detection"""
        try:
            # Detect encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except Exception as e:
            print(f"⚠️  Error reading text file {file_path}: {str(e)}")
            return None
    
    def compress_text(self, text):
        """Apply compression based on level"""
        if not text:
            return ""
        
        level = self.config['compression_level']
        
        if level == 1:  # Light
            # Remove multiple newlines
            text = re.sub(r'\n{3,}', '\n\n', text)
            # Remove trailing spaces
            text = re.sub(r' +$', '', text, flags=re.MULTILINE)
        
        elif level == 2:  # Medium
            # All from light plus:
            text = re.sub(r'\n{3,}', '\n\n', text)
            text = re.sub(r' +$', '', text, flags=re.MULTILINE)
            # Normalize whitespace
            text = re.sub(r' {2,}', ' ', text)
            # Remove empty lines with only whitespace
            text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n{2,}', '\n\n', text)
        
        elif level == 3:  # Heavy
            # All from medium plus:
            text = re.sub(r'\s+', ' ', text)
            # Keep paragraph breaks
            text = re.sub(r'\.(\s+)([A-Z])', '.\n\n\\2', text)
            # Remove redundant punctuation
            text = re.sub(r'([.!?])\1+', r'\1', text)
        
        return text.strip()
    
    def chunk_text(self, text, file_info):
        """Intelligently chunk text while preserving context"""
        if not text:
            return []
        
        # Use tiktoken for accurate token counting
        encoding = tiktoken.get_encoding("cl100k_base")
        
        chunks = []
        sentences = nltk.sent_tokenize(text)
        
        current_chunk = ""
        current_tokens = 0
        
        for sentence in sentences:
            sentence_tokens = len(encoding.encode(sentence))
            
            if current_tokens + sentence_tokens > self.config['chunk_size']:
                if current_chunk:
                    chunks.append({
                        'content': current_chunk.strip(),
                        'tokens': current_tokens,
                        'chunk_index': len(chunks),
                        'file_info': file_info
                    })
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # Add remaining chunk
        if current_chunk:
            chunks.append({
                'content': current_chunk.strip(),
                'tokens': current_tokens,
                'chunk_index': len(chunks),
                'file_info': file_info
            })
        
        return chunks
    
    def process_file(self, file_path):
        """Process a single file"""
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_name)[1].lower()
        
        print(f"📄 Processing: {file_name}")
        
        # Get file metadata
        file_stat = os.stat(file_path)
        file_info = {
            'file_name': file_name,
            'file_path': str(file_path),
            'file_size': file_stat.st_size,
            'modified_time': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
            'file_hash': self.get_file_hash(file_path)
        }
        
        # Extract text based on file type
        if file_ext == '.pdf':
            text, num_pages = self.extract_text_from_pdf(file_path)
            if text:
                file_info['num_pages'] = num_pages
                file_info['file_type'] = 'pdf'
                self.stats['pdf_files'] += 1
        elif file_ext in ['.txt', '.text', '.md', '.markdown']:
            text = self.extract_text_from_file(file_path)
            if text:
                file_info['file_type'] = 'text'
                self.stats['text_files'] += 1
        else:
            print(f"⚠️  Skipping unsupported file type: {file_ext}")
            return None
        
        if not text:
            self.stats['failed_files'].append(file_name)
            return None
        
        # Track original size
        original_size = len(text)
        self.stats['total_chars_before'] += original_size
        
        # Compress text
        compressed_text = self.compress_text(text)
        compressed_size = len(compressed_text)
        self.stats['total_chars_after'] += compressed_size
        
        # Add compression info
        file_info['original_chars'] = original_size
        file_info['compressed_chars'] = compressed_size
        file_info['compression_ratio'] = round((1 - compressed_size/original_size) * 100, 2) if original_size > 0 else 0
        
        # Chunk the compressed text
        chunks = self.chunk_text(compressed_text, file_info)
        
        return {
            'file_info': file_info,
            'chunks': chunks,
            'total_chunks': len(chunks)
        }
    
    def get_file_hash(self, file_path):
        """Generate file hash for tracking"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def process_directory(self):
        """Process all files in directory"""
        input_dir = Path(self.config['input_dir'])
        all_files = list(input_dir.rglob('*.pdf')) + list(input_dir.rglob('*.txt')) + \
                   list(input_dir.rglob('*.text')) + list(input_dir.rglob('*.md')) + \
                   list(input_dir.rglob('*.markdown'))
        
        self.stats['total_files'] = len(all_files)
        
        if not all_files:
            print("❌ No PDF or text files found in the directory!")
            return None
        
        print(f"\n🔍 Found {len(all_files)} files to process\n")
        
        compressed_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'source_directory': str(input_dir),
                'compression_config': self.config,
                'total_files': len(all_files)
            },
            'documents': []
        }
        
        for i, file_path in enumerate(all_files, 1):
            print(f"\n[{i}/{len(all_files)}] ", end="")
            result = self.process_file(file_path)
            if result:
                compressed_data['documents'].append(result)
        
        # Add statistics
        compressed_data['statistics'] = self.stats
        
        return compressed_data
    
    def save_compressed_data(self, data):
        """Save compressed data to file"""
        output_path = os.path.join(os.path.dirname(self.config['input_dir']), self.config['output_file'])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Compressed data saved to: {output_path}")
        
        # Print statistics
        print("\n=== Compression Statistics ===")
        print(f"📊 Total files processed: {self.stats['total_files']}")
        print(f"📑 PDF files: {self.stats['pdf_files']}")
        print(f"📝 Text files: {self.stats['text_files']}")
        print(f"❌ Failed files: {len(self.stats['failed_files'])}")
        
        if self.stats['total_chars_before'] > 0:
            compression_ratio = (1 - self.stats['total_chars_after'] / self.stats['total_chars_before']) * 100
            print(f"\n🗜️  Overall compression ratio: {compression_ratio:.1f}%")
            print(f"📏 Characters before: {self.stats['total_chars_before']:,}")
            print(f"📐 Characters after: {self.stats['total_chars_after']:,}")
        
        # File size
        file_size = os.path.getsize(output_path)
        print(f"\n💾 Output file size: {file_size / 1024 / 1024:.2f} MB")
        
        if self.stats['failed_files']:
            print(f"\n⚠️  Failed files: {', '.join(self.stats['failed_files'])}")
    
    def run(self):
        """Main execution flow"""
        print("\n🚀 Document Compressor for AI v1.0")
        print("=" * 40)
        
        if not self.get_user_config():
            print("\n❌ Configuration cancelled.")
            return
        
        print("\n🏃 Starting compression process...")
        
        compressed_data = self.process_directory()
        
        if compressed_data:
            self.save_compressed_data(compressed_data)
            print("\n✨ Compression complete!")
            
            # Provide usage instructions
            print("\n📖 How to use with Claude or other AIs:")
            print("1. The compressed file contains all documents in digestible chunks")
            print("2. Each chunk includes file metadata for context")
            print("3. You can feed chunks sequentially or search for specific files")
            print("4. All essential information is preserved from the original documents")
        else:
            print("\n❌ No data was processed.")

def main():
    """Entry point"""
    compressor = DocumentCompressor()
    try:
        compressor.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
    except Exception as e:
        print(f"\n❌ An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
