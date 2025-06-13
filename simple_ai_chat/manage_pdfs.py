import os
import shutil
import argparse
from pathlib import Path

def add_pdf_files(file_paths):
    """Add PDF files to the data/pdfs folder"""
    # Create the PDF folder if it doesn't exist
    pdf_folder = os.path.join("data", "pdfs")
    os.makedirs(pdf_folder, exist_ok=True)
    
    added_files = []
    for file_path in file_paths:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue
            
        # Check if file is a PDF
        if not file_path.lower().endswith('.pdf'):
            print(f"Not a PDF file: {file_path}")
            continue
            
        # Get filename and destination path
        filename = os.path.basename(file_path)
        dest_path = os.path.join(pdf_folder, filename)
        
        # Copy file
        try:
            shutil.copy2(file_path, dest_path)
            added_files.append(dest_path)
            print(f"Added: {filename}")
        except Exception as e:
            print(f"Error adding {filename}: {e}")
    
    return added_files

def list_pdf_files():
    """List all PDF files in the data/pdfs folder"""
    pdf_folder = os.path.join("data", "pdfs")
    
    if not os.path.exists(pdf_folder):
        print(f"PDF folder not found: {pdf_folder}")
        return []
        
    pdf_files = [f for f in os.listdir(pdf_folder) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("No PDF files found in data/pdfs folder.")
    else:
        print(f"Found {len(pdf_files)} PDF files:")
        for i, file in enumerate(pdf_files, 1):
            file_path = os.path.join(pdf_folder, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"{i}. {file} ({file_size:.1f} KB)")
    
    return pdf_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage PDF files for the AI Chat application")
    parser.add_argument("--add", nargs='+', help="Path(s) to PDF file(s) to add")
    parser.add_argument("--list", action="store_true", help="List all PDF files in the data/pdfs folder")
    
    args = parser.parse_args()
    
    if args.add:
        added = add_pdf_files(args.add)
        if added:
            print(f"\nSuccessfully added {len(added)} PDF file(s).")
            print("You can now run `python main.py process` to process the files.")
    elif args.list:
        list_pdf_files()
    else:
        parser.print_help()
