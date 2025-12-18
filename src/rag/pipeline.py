import torch
from transformers import pipeline
from .database import CookingDB

class RAGPipeline:
    def __init__(self, db_path="./data/chroma_db"):
        self.db = CookingDB(db_path)
        self.llm = self._load_model()
        
    def _load_model(self):
        print("Loading TinyLlama model...")
        # Check for GPU
        if torch.cuda.is_available():
            device = 0 
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
            
        return pipeline(
            'text-generation',
            model='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            device=device
        )

    def answer_question(self, question):
        # 1. Retrieve
        results = self.db.query(question, n_results=3)
        if not results['documents'][0]:
            return "I couldn't find any relevant cooking info in my database."
            
        contexts = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context_text = "\n\n---\n\n".join(contexts)
        
        # 2. Augmented Generation
        prompt = f"""<|system|>
        You are a helpful cooking assistant. Answer questions based on the provided context. Be concise.<|end|>
        <|user|>
        Context:
        {context_text}

        Question: {question}<|end|>
        <|assistant|>"""

        output = self.llm(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_full_text=False
        )
        
        answer = output[0]['generated_text'].strip()
        
        # Format sources for display
        sources = list(set([m.get('source', 'unknown').split('/')[-1] for m in metadatas]))
        
        return answer, sources

# Initialize a singleton instance for the app to import
rag_pipeline = RAGPipeline()

def answer_question(query):
    """Simple wrapper for the Streamlit app"""
    answer, sources = rag_pipeline.answer_question(query)
    return f"{answer}\n\n**Sources:** {', '.join(sources)}"