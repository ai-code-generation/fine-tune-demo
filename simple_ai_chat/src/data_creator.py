import json
import random
from typing import List, Dict

class DatasetCreator:
    """Create training dataset from processed documents"""
    
    def __init__(self):
        self.question_templates = [
            "Summarize content about {}",
            "Explain about {}",
            "Provide details about {}",
            "Talk about {}",
            "Describe {}",
        ]
    def extract_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text"""
        # Simple: get words longer than 4 characters
        words = text.split()
        keywords = [word for word in words 
                   if len(word) > 4 and word.isalpha()]
        
        # If no keywords match the condition, try any word
        if not keywords and words:
            keywords = [word for word in words if word.isalpha()]
            
        return list(set(keywords))[:10]  # Top 10 keywords
    def generate_qa_from_chunk(self, chunk: str) -> List[Dict]:
        """Generate Q&A pairs from text chunk"""
        qa_pairs = []
        keywords = self.extract_keywords(chunk)
        if not keywords:
            # If no keywords found, create a default Q&A pair
            qa_pairs.append({
                "input": "Please summarize this text",
                "output": chunk,
                "context": f"Based on document: {chunk[:100]}..."
            })
            return qa_pairs
            
        for keyword in keywords[:3]:  # 3 Q&A per chunk
            question = random.choice(self.question_templates).format(keyword)
            qa_pairs.append({
                "input": question,
                "output": chunk,
                "context": f"Based on document: {chunk[:100]}..."
            })
        
        return qa_pairs
    def create_training_dataset(self, documents: List[Dict]) -> List[Dict]:
        """Create complete training dataset"""
        training_data = []
        
        for doc in documents:
            # Split document into chunks
            chunks = self.chunk_text(doc['content'])
            
            for chunk in chunks:
                qa_pairs = self.generate_qa_from_chunk(chunk)
                training_data.extend(qa_pairs)
                
            print(f"Generated {len(qa_pairs)} Q&A pairs from {doc['filename']}")
        
        # Limit dataset size if configured
        max_examples = 1000
        if len(training_data) > max_examples:
            print(f"Limiting dataset from {len(training_data)} to {max_examples} examples")
            training_data = random.sample(training_data, max_examples)
            
        return training_data
    def chunk_text(self, text: str, chunk_size: int = 300) -> List[str]:
        """Split text into chunks"""
        sentences = text.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + "."
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + "."
        
        if current_chunk:
            chunks.append(current_chunk.strip())
            
        return chunks
    
    def save_dataset(self, dataset: List[Dict], output_path: str):
        """Save dataset to JSON file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Saved dataset with {len(dataset)} examples to {output_path}")
