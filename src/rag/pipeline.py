import torch
from transformers import pipeline
from .database import CookingDB

class RAGPipeline:
    def __init__(self, db_path="./data/chroma_db"):
        self.db = CookingDB(db_path)
        self.llm = self._load_model()
        self.conversation_history = []  # Store conversation turns
        
    def _load_model(self):
        print("Loading Phi-3-mini model...")
        
        # Check for device
        if torch.cuda.is_available():
            device = 0 
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
            
        return pipeline(
            'text-generation',
            model='microsoft/Phi-3-mini-4k-instruct',
            device=device,
            torch_dtype=torch.float16,  # Use FP16 for efficiency
            model_kwargs={"attn_implementation": "eager"}  # Better for MPS
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
        """Main entry point - routes to simple or reasoning-based answer"""
        
        # Quick classification: Does this need multi-step reasoning?
        needs_reasoning = any(word in question.lower() for word in [
            'why', 'how', 'explain', 'what happens', 'difference', 'compare',
            'should i', 'can i', 'substitute', 'instead of', 'alternative',
            'better', 'versus', 'vs', 'or'
        ])
        
        if needs_reasoning:
            print("🧠 Using multi-step reasoning...")
            return self._answer_with_reasoning(question)
        else:
            print("⚡ Using simple retrieval...")
            return self._answer_simple(question)

    def _answer_simple(self, question):
        """Single-pass answer for simple questions"""
        
        # 1. Rewrite query with conversation context
        contextualized_query = self._rewrite_query_with_context(question)
        
        # 2. Retrieve using contextualized query
        results = self.db.query(contextualized_query, n_results=3)
        if not results['documents'][0]:
            return "I couldn't find any relevant cooking info in my database.", []
            
        contexts = results['documents'][0]
        metadatas = results['metadatas'][0]
        
        context_text = "\n\n---\n\n".join(contexts)
        
        # Truncate if too long
        if len(context_text) > 1200:
            context_text = context_text[:1200] + "..."
        
        # 3. Generate answer
        answer = self._generate_answer(question, context_text)
        
        # 4. Store in conversation history
        self.conversation_history.append({
            'user': question,
            'assistant': answer
        })
        
        # Keep only last 10 turns
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Format sources
        sources = list(set([m.get('source', 'unknown').split('/')[-1] for m in metadatas]))
        
        return answer, sources

    def _answer_with_reasoning(self, question):
        """Two-step reasoning for complex questions"""
        
        # STEP 1: Break down the question
        breakdown_prompt = f"Break this cooking question into 2 simple sub-questions:\n\nQuestion: {question}\n\nSub-questions:\n1."
        
        try:
            breakdown = self.llm(
                breakdown_prompt,
                max_new_tokens=50,
                do_sample=False,
                return_full_text=False,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )[0]['generated_text'].strip()
            
            # Extract sub-questions (simple parsing)
            sub_questions = [q.strip() for q in breakdown.split('\n') if q.strip() and any(c.isalpha() for c in q)][:2]
            
            if len(sub_questions) < 1:
                # Fallback: treat as simple question
                print("⚠️ Breakdown failed, falling back to simple...")
                return self._answer_simple(question)
                
        except Exception as e:
            print(f"⚠️ Breakdown error: {e}, falling back to simple...")
            return self._answer_simple(question)
        
        # STEP 2: Retrieve context for each sub-question
        all_contexts = []
        all_metadatas = []
        
        for sub_q in sub_questions:
            results = self.db.query(sub_q, n_results=2)  # Only 2 per sub-question
            if results['documents'][0]:
                all_contexts.extend(results['documents'][0])
                all_metadatas.extend(results['metadatas'][0])
        
        if not all_contexts:
            return "I couldn't find relevant information.", []
        
        # Deduplicate contexts
        unique_contexts = list(dict.fromkeys(all_contexts))[:3]  # Max 3 total
        context_text = "\n\n---\n\n".join(unique_contexts)
        
        # Truncate if too long
        if len(context_text) > 1200:
            context_text = context_text[:1200] + "..."
        
        # STEP 3: Answer with combined context
        answer = self._generate_answer(question, context_text)
        
        # Store in history
        self.conversation_history.append({
            'user': question,
            'assistant': answer
        })
        
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        sources = list(set([m.get('source', 'unknown').split('/')[-1] for m in all_metadatas]))
        
        return answer, sources

    def _generate_answer(self, question, context_text):
        """Generate answer from question and context with error handling"""
        
        # Build prompt
        prompt = f"""<|system|>
You are a helpful cooking assistant. Answer the user's question using ONLY the context provided below. Be conversational and friendly.

CRITICAL RULES:
- If the context doesn't contain the answer, say "I don't have that information in my knowledge base"
- Keep answers under 150 words
- Write naturally, like talking to a friend
- Focus on practical, actionable advice
<|end|>

<|user|>
Context:
{context_text}

Question: {question}
<|end|>

<|assistant|>"""

        # Try sampling first, fallback to greedy on error
        try:
            output = self.llm(
                prompt,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.1,
                return_full_text=False,
                eos_token_id=self.llm.tokenizer.eos_token_id,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
        except RuntimeError as e:
            if "probability" in str(e) or "inf" in str(e) or "nan" in str(e):
                print("⚠️ Sampling failed, falling back to greedy decoding...")
                output = self.llm(
                    prompt,
                    max_new_tokens=300,
                    do_sample=False,
                    return_full_text=False,
                    pad_token_id=self.llm.tokenizer.eos_token_id
                )
            else:
                raise
        
        answer = output[0]["generated_text"]
        
        # Clean up answer - remove any special tokens
        for token in ["<|assistant|>", "<|user|>", "<|end|>", "<|system|>", "</s>"]:
            if token in answer:
                answer = answer.split(token)[0]
        
        # Remove partial token fragments
        if "<|" in answer:
            answer = answer.split("<|")[0]
        
        # Remove common prefixes
        answer = answer.lstrip().removeprefix("Response:").removeprefix("Answer:").strip()
        
        return answer

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        return "Conversation history cleared."

# Initialize a singleton instance for the app to import
rag_pipeline = RAGPipeline()