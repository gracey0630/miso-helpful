import torch
from transformers import pipeline
from .database import CookingDB

class RAGPipeline:
    def __init__(self, db_path="./data/chroma_db"):
        self.db = CookingDB(db_path)
        self.llm = self._load_model()
        self.conversation_history = []  # Store conversation turns
        
    def _load_model(self):
        print("Loading Qwen2.5-3B model...")
    
        if torch.cuda.is_available():
            device = 0 
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = -1
            
        return pipeline(
            'text-generation',
            model='Qwen/Qwen2.5-3B-Instruct',
            device=device,
            torch_dtype=torch.float16
        )

    def _rewrite_query_with_context(self, question):
        """Rewrite the current question using conversation history only if relevant"""
        if not self.conversation_history:
            return question
        
        # Check if question contains reference words (pronouns, "it", "that", etc.)
        reference_words = ['it', 'that', 'this', 'them', 'those', 'the same', 'also', 'too']
        has_reference = any(word in question.lower() for word in reference_words)
        
        # Check if question is very short (likely a follow-up)
        is_short = len(question.split()) < 5
        
        # Only use context if there are clear references or very short question
        if not (has_reference or is_short):
            print("🔄 New topic detected, ignoring previous context")
            return question
        
        # Get last 2 turns for context
        recent_history = self.conversation_history[-2:]
        history_text = "\n".join([
            f"User: {turn['user']}\nAssistant: {turn['assistant'][:100]}"  # Truncate assistant
            for turn in recent_history
        ])
        
        contextualized = f"Given previous discussion:\n{history_text}\n\nCurrent question: {question}"
        print("🔗 Using conversation context")
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
        
        # Rewrite query with conversation context
        contextualized_query = self._rewrite_query_with_context(question)
        
        # STEP 1: Break down the question (use contextualized query)
        breakdown_prompt = f"Break this cooking question into 2 simple sub-questions:\n\nQuestion: {contextualized_query}\n\nSub-questions:\n1."
        
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
                print("⚠️ Breakdown failed, falling back to simple...")
                return self._answer_simple(question)
                
        except Exception as e:
            print(f"⚠️ Breakdown error: {e}, falling back to simple...")
            return self._answer_simple(question)
        
        # STEP 2: Retrieve context for each sub-question
        all_contexts = []
        all_metadatas = []
        
        for sub_q in sub_questions:
            results = self.db.query(sub_q, n_results=2)
            if results['documents'][0]:
                all_contexts.extend(results['documents'][0])
                all_metadatas.extend(results['metadatas'][0])
        
        if not all_contexts:
            return "I couldn't find relevant information.", []
        
        # Deduplicate contexts
        unique_contexts = list(dict.fromkeys(all_contexts))[:3]
        context_text = "\n\n---\n\n".join(unique_contexts)
        
        if len(context_text) > 1200:
            context_text = context_text[:1200] + "..."
        
        # STEP 3: Answer with combined context (use original question for answer generation)
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
- Do not repeat rules below in your response.
- If the context doesn't contain the answer, say "I don't have that information in my knowledge base".
- Keep answers under 150 words.
- Responses should be natural and conversational speech.
- Focus on practical, actionable advice.
- Prioritize the "Context from knowledge base" for facts.
- Use **bold** for ingredients.
- Use numbered lists for instructions.
- If the previous contexts are irrelevent to the new query, ignore previous contexts.
- ALWAYS end your response with a complete sentence and a closing remark like "Enjoy your meal!" or "Happy cooking!".
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
                max_new_tokens=200,           # REDUCED from 300
                do_sample=True,
                temperature=0.7,
                top_p=0.7,                   # REDUCED from 0.9 (more focused)
                top_k=40,                     # REDUCED from 50 (less randomness)
                repetition_penalty=1.2,       # INCREASED from 1.1 (penalize repetition more)
                no_repeat_ngram_size=3,       # ADD THIS (prevents repeating 3-word sequences)
                return_full_text=False,
                eos_token_id=self.llm.tokenizer.eos_token_id,
                pad_token_id=self.llm.tokenizer.eos_token_id
            )
        except RuntimeError as e:
            if "probability" in str(e) or "inf" in str(e) or "nan" in str(e):
                print("⚠️ Sampling failed, falling back to greedy decoding...")
                output = self.llm(
                    prompt,
                    max_new_tokens=200,       # REDUCED
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