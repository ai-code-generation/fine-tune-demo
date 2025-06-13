"""
Script để chuyển file text thành PDF để test ứng dụng

Yêu cầu:
- pip install fpdf
"""

from fpdf import FPDF
import os
import argparse

def text_to_pdf(text_file, pdf_file):
    """Chuyển file text sang PDF đơn giản"""
    try:
        # Kiểm tra file text có tồn tại không
        if not os.path.exists(text_file):
            print(f"File text không tồn tại: {text_file}")
            return False
        
        # Đọc nội dung file text
        with open(text_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Tạo PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Hỗ trợ tiếng Việt, thêm font Unicode
        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        pdf.set_font('DejaVu', '', 12)
        
        # Thêm nội dung vào PDF
        for line in content.split('\n'):
            pdf.multi_cell(0, 10, line)
        
        # Lưu file PDF
        pdf.output(pdf_file)
        print(f"Đã tạo file PDF: {pdf_file}")
        return True
        
    except Exception as e:
        print(f"Lỗi khi tạo PDF: {e}")
        return False

def create_sample_pdf():
    """Tạo sample PDF từ file text mẫu"""
    # Đường dẫn tới thư mục data/pdfs
    pdf_folder = os.path.join("data", "pdfs")
    
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(pdf_folder, exist_ok=True)
    
    # Đường dẫn tới file text và PDF
    text_file = "sample_document.txt"
    pdf_file = os.path.join(pdf_folder, "sample_document.pdf")
    
    if text_to_pdf(text_file, pdf_file):
        print(f"""
Đã tạo file PDF mẫu tại: {pdf_file}
Bây giờ bạn có thể chạy:
    python main.py process
để xử lý file PDF và tạo training data.
        """)
    else:
        print("""
Lỗi khi tạo file PDF mẫu. Vui lòng cài đặt thư viện fpdf:
    pip install fpdf
và tải font DejaVuSansCondensed.ttf để hỗ trợ tiếng Việt.
        """)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chuyển file text sang PDF")
    parser.add_argument("--text", type=str, default="sample_document.txt",
                        help="Đường dẫn tới file text")
    parser.add_argument("--output", type=str, default="data/pdfs/output.pdf",
                        help="Đường dẫn tới file PDF output")
    parser.add_argument("--sample", action="store_true",
                        help="Tạo file PDF mẫu từ sample_document.txt")
    
    args = parser.parse_args()
    
    if args.sample:
        create_sample_pdf()
    else:
        text_to_pdf(args.text, args.output)
