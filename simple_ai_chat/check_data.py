"""
Công cụ kiểm tra dữ liệu trước khi huấn luyện
"""

import json
import os
from transformers import AutoTokenizer
import argparse
from datasets import Dataset

def check_training_data(json_path="data/training_data.json", model_name="gpt2"):
    """Kiểm tra dữ liệu huấn luyện và tiến hành tokenize thử"""
    
    # Kiểm tra file tồn tại
    if not os.path.exists(json_path):
        print(f"❌ File dữ liệu không tồn tại: {json_path}")
        return
    
    # Đọc dữ liệu
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✓ Đã đọc {len(data)} mẫu dữ liệu từ {json_path}")
    except Exception as e:
        print(f"❌ Lỗi khi đọc file JSON: {e}")
        return
    
    # Kiểm tra cấu trúc
    valid_items = 0
    invalid_items = 0
    
    for i, item in enumerate(data[:10]):  # Chỉ kiểm tra 10 mẫu đầu tiên
        if 'input' in item and 'output' in item:
            valid_items += 1
            print(f"\n✓ Mẫu #{i} hợp lệ:")
            print(f"  - Input: {item['input'][:50]}...")
            print(f"  - Output: {item['output'][:50]}...")
        else:
            invalid_items += 1
            print(f"\n❌ Mẫu #{i} không hợp lệ:")
            print(f"  - Item: {item}")
    
    # Thống kê
    print(f"\nTổng kết: {valid_items} mẫu hợp lệ, {invalid_items} mẫu không hợp lệ trong 10 mẫu đầu")
    
    # Format dữ liệu để tokenize thử
    print("\n🔄 Thử tokenize dữ liệu...")
    
    try:
        # Tạo tokenizer
        print(f"Đang tải tokenizer {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Thêm padding token nếu không có
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tạo dataset từ list
        formatted_data = []
        for item in data[:5]:  # Chỉ thử 5 mẫu đầu
            text = f"Human: {item['input']}\nAssistant: {item['output']}"
            formatted_data.append({"text": text})
        
        # In ra một mẫu dữ liệu đã format
        print("\nMẫu dữ liệu đã format:")
        print(formatted_data[0]["text"][:100] + "...")
        
        # Tạo dataset
        dataset = Dataset.from_list(formatted_data)
        print(f"\n✓ Đã tạo dataset với {len(dataset)} mẫu và các cột: {dataset.column_names}")
        
        # Tokenize thử
        print("\n🔄 Thử tokenize...")
        
        # Định nghĩa hàm tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )
        
        # Map với batched=False trước
        print("Thử tokenize không dùng batched...")
        result1 = dataset.map(tokenize_function, batched=False)
        print(f"✓ Tokenize thành công! Cột trong dataset: {result1.column_names}")
        
        # Map với batched=True
        print("\nThử tokenize với batched=True...")
        result2 = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        print(f"✓ Tokenize batched thành công! Cột trong dataset: {result2.column_names}")
        
        print("\n✓ Kiểm tra hoàn tất! Dữ liệu của bạn có thể được sử dụng cho huấn luyện.")
        print("Để tiếp tục huấn luyện, hãy chạy: python main.py train")
        
    except Exception as e:
        print(f"\n❌ Lỗi khi tokenize: {e}")
        print("Vui lòng kiểm tra và sửa dữ liệu trước khi huấn luyện.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kiểm tra dữ liệu huấn luyện")
    parser.add_argument("--data", type=str, default="data/training_data.json",
                       help="Đường dẫn tới file dữ liệu JSON")
    parser.add_argument("--model", type=str, default="gpt2",
                       help="Tên model để tải tokenizer")
    
    args = parser.parse_args()
    
    check_training_data(args.data, args.model)
