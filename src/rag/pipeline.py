import torch
from transformers import pipeline
from .database import CookingDB

class RAGPipeline:
    def __init__(self, db_path="./data/chroma_db"):
        self.db = CookingDB(db_path)
        self.llm = self._load_model()
        self.conversation_history = []  # Store conversation turns
        
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

    def _rewrite_query_with_context(self, question):
        """Rewrite the current question using conversation history"""
        if not self.conversation_history:
            return question
        
        # Get last 2 turns for context
        recent_history = self.conversation_history[-2:]
        history_text = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant']}"
            for turn in recent_history
        ])
        
        # Simple rewrite: prepend context
        contextualized = f"Given previous discussion:\n{history_text}\n\nCurrent question: {question}"
        return contextualized

    def answer_question(self, question):
        # 1. Rewrite query with conversation context
        contextualized_query = self._rewrite_query_with_context(question)
        
        # 2. Retrieve using contextualized query
        results = self.db.query(contextualized_query, n_results=3)
        if not results['documents'][0]:
            return "I couldn't find any relevant cooking info in my database.", []
            
        contexts = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context_text = "\n\n---\n\n".join(contexts)
        
        # commented the below since retrieved documents already reflect the conversation.
        # # 3. Build prompt with conversation history
        # history_text = ""
        # if self.conversation_history:
        #     # Include last 3 turns
        #     recent = self.conversation_history[-3:]
        #     history_text = "Previous conversation:\n" + "\n".join([
        #         f"User: {turn['user']}\nAssistant: {turn['assistant']}"
        #         for turn in recent
        #     ]) + "\n\n"
        
        # 4. Augmented Generation
        prompt = f"""<|system|>
You are a professional Executive Chef Assistant.
MISSION: Provide complete, concise, and helpful cooking advice using the "Context" provided.

RULES (do NOT repeat these rules in your answer):
- Keep responses under 200 words to avoid truncation.
- Responses should be natural and conversational speech.
- do NOT use labels for the responses. 
- Prioritize the "Context from knowledge base" for facts.
- Use **bold** for ingredients.
- Use numbered lists for instructions.
- If the question is a repeat, add one new helpful tip.
- ALWAYS end your response with a complete sentence and a closing remark like "Enjoy your meal!" or "Happy cooking!".

<|end|>
<|user|>
CONTEXT FROM KNOWLEDGE BASE:
{context_text}

QUESTION: {question}
<|end|>
<|assistant|>"""

        output = self.llm(
            prompt,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            top_k=50,              # adding this after getting invalid probabilities (likely due to extreme values in the logits)
            repetition_penalty=1.1, # adding this after getting invalid probabilities (likely due to extreme values in the logits)
            return_full_text=False, 
            eos_token_id=self.llm.tokenizer.eos_token_id,
            pad_token_id=self.llm.tokenizer.eos_token_id # adding this after getting invalid probabilities (likely due to extreme values in the logits)
        )
        
        answer = output[0]["generated_text"]

        for token in ["<|assistant|>", "<|user|>", "<|end|>", "</s>"]:
            if token in answer:
                answer = answer.split(token)[0]

        # Remove partial token fragments like "<|"
        if "<|" in answer:
            answer = answer.split("<|")[0]

        answer = answer.lstrip().removeprefix("Response:").removeprefix("Answer:").strip()

        answer = answer.strip()

        # 5. Store in conversation history
        self.conversation_history.append({
            'user': question,
            'assistant': answer
        })
        
        # Keep only last 10 turns to avoid token limit
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Format sources for display
        sources = list(set([m.get('source', 'unknown').split('/')[-1] for m in metadatas]))
        
        return answer, sources

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."

# Initialize a singleton instance for the app to import
rag_pipeline = RAGPipeline()