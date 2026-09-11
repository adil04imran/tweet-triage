import os
import json
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

class AppleSupportAgent:
    def __init__(self, intents_path, db_path):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        
        self.client = Groq(api_key=api_key)
        self.model_name = "qwen/qwen3.8-27b"
        
        # Load Intents Taxonomy
        with open(intents_path, 'r') as f:
            self.intents = json.load(f)
            
        # Initialize RAG Retriever
        self.db_client = chromadb.PersistentClient(path=db_path)
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.collection = self.db_client.get_collection(
            name="apple_support_replies",
            embedding_function=self.sentence_transformer_ef
        )
        
    def classify_intent(self, text):
        intents_desc = "\n".join([f"- {i['intent_name']}: {i['description']}" for i in self.intents])
        
        prompt = f"""
        You are an AI support classifier for Apple.
        Given the following customer message, categorize it into exactly ONE of the following intents.
        
        Available Intents:
        {intents_desc}
        
        Customer Message: "{text}"
        
        Respond with ONLY the exact intent_name string. No markdown, no quotes, nothing else.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return "general_help" # fallback
            
    def retrieve_context(self, text, k=3):
        results = self.collection.query(
            query_texts=[text],
            n_results=k
        )
        # return list of historical brand replies
        if results and results['metadatas'] and len(results['metadatas'][0]) > 0:
            return [meta['brand_reply'] for meta in results['metadatas'][0]]
        return []

    def draft_reply(self, text, intent, context_replies):
        context_str = "\n".join([f"- {reply}" for reply in context_replies])
        
        prompt = f"""
        You are an official Apple Support representative on Twitter.
        The customer has tweeted: "{text}"
        The issue has been classified as: {intent}
        
        Historically, Apple Support resolved similar issues with these replies:
        {context_str}
        
        Draft a short, empathetic, and helpful reply (under 280 characters). Do not invent links. 
        If a link is needed, use a generic placeholder like https://apple.co/support.
        
        Reply:
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.7
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return "We're here to help. Please DM us more details so we can take a closer look."
            
    def should_escalate(self, text, intent, drafted_reply):
        # 1. Rule-based escalation
        lower_text = text.lower()
        if any(word in lower_text for word in ['sue', 'lawyer', 'refund', 'manager']):
            return True, "Customer used high-risk keywords (legal/refund)."
            
        # 2. LLM Judgement
        prompt = f"""
        Analyze this customer support interaction:
        Customer: "{text}"
        Drafted Reply: "{drafted_reply}"
        
        Does this issue require a human agent? Reply 'YES' or 'NO' followed by a short reason.
        Escalate if:
        - The customer is extremely angry.
        - The issue requires sensitive account access (PII).
        - The drafted reply seems unhelpful or generic.
        """
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                temperature=0.0
            )
            resp_text = chat_completion.choices[0].message.content.strip()
            
            if resp_text.upper().startswith("YES"):
                reason = resp_text[3:].strip(":- ")
                return True, reason
            return False, "Handled safely by auto-reply."
        except Exception as e:
            return True, "Fallback to human due to LLM error."
            
    def handle_ticket(self, text):
        intent = self.classify_intent(text)
        context = self.retrieve_context(text)
        reply = self.draft_reply(text, intent, context)
        escalate, reason = self.should_escalate(text, intent, reply)
        
        return {
            "intent": intent,
            "drafted_reply": reply,
            "should_escalate": escalate,
            "escalation_reason": reason,
            "retrieved_context": context
        }

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test script
    agent = AppleSupportAgent(
        intents_path="../../data/processed/discovered_intents.json",
        db_path="../../data/chroma_db"
    )
    
    test_msg = "My battery on the iphone 6s is draining 10x faster after the new iOS update. Please help me!!!"
    print(f"Testing with: {test_msg}")
    result = agent.handle_ticket(test_msg)
    print(json.dumps(result, indent=2))
