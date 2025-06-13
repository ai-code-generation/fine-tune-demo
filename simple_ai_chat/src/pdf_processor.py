import PyPDF2
import os
from typing import List, Dict

class PDFProcessor:
    """Xử lý PDF files từ folder input"""
    
    def __init__(self, pdf_folder: str):
        self.pdf_folder = pdf_folder
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text từ 1 file PDF"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {e}")
            return ""
    
    def process_all_pdfs(self) -> List[Dict]:
        """Process tất cả PDF trong folder"""
        documents = []
        pdf_files = [f for f in os.listdir(self.pdf_folder) if f.endswith('.pdf')]
        
        if not pdf_files:
            print(f"No PDF files found in {self.pdf_folder}")
            return documents
            
        for filename in pdf_files:
            pdf_path = os.path.join(self.pdf_folder, filename)
            print(f"Processing: {filename}")
            text = self.extract_text_from_pdf(pdf_path)
            if text:
                documents.append({
                    'filename': filename,
                    'content': self.clean_text(text)
                })
        return documents
    
    def clean_text(self, text: str) -> str:
        """Làm sạch text cơ bản"""
        # Loại bỏ newlines thừa, ký tự đặc biệt
        text = text.replace('\n', ' ').replace('\t', ' ')
        # Loại bỏ khoảng trắng thừa
        text = ' '.join(text.split())
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Chia text thành chunks nhỏ"""
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        return chunks
