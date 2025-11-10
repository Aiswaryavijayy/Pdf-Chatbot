import os
from pymongo import MongoClient
from dotenv import load_dotenv
from difflib import SequenceMatcher
import re
from datetime import datetime
from openai import OpenAI

load_dotenv()

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["pdf_qa"]
collection = db["chat_history"]

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def save_chat(question: str, answer: str, is_followup: bool = False, parent_question: str = None, chat_id: str = None):
    """Store question and answer in MongoDB with chat_id"""
    try:
        chat_data = {
            "question": question,
            "answer": answer,
            "is_followup": is_followup,
            "timestamp": datetime.now(),
            "chat_id": chat_id
        }
        if parent_question:
            chat_data["parent_question"] = parent_question
        collection.insert_one(chat_data)
    except Exception as e:
        print(f"Error saving chat: {e}")

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    if not text:
        return ""
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts"""
    if not text1 or not text2:
        return 0.0
    norm_text1 = normalize_text(text1)
    norm_text2 = normalize_text(text2)
    return SequenceMatcher(None, norm_text1, norm_text2).ratio()

def check_similar_question(question: str, chat_id: str, similarity_threshold: float = 0.8):
    """Check for similar questions within the same chat_id - increased threshold for better precision"""
    try:
        all_chats = list(collection.find({"chat_id": chat_id}))
        best_match = None
        best_similarity = 0
        
        # Normalize the input question for better comparison
        normalized_question = normalize_text(question)
        
        for chat in all_chats:
            stored_question = chat.get("question", "")
            if stored_question:
                # Check both original and normalized similarity
                similarity = calculate_similarity(question, stored_question)
                normalized_similarity = calculate_similarity(normalized_question, normalize_text(stored_question))
                
                # Use the higher of the two similarities
                final_similarity = max(similarity, normalized_similarity)
                
                if final_similarity > best_similarity and final_similarity >= similarity_threshold:
                    best_similarity = final_similarity
                    best_match = chat
        
        if best_match:
            return {
                "found": True,
                "answer": best_match.get("answer", ""),
                "original_question": best_match.get("question", ""),
                "similarity": best_similarity,
                "_id": str(best_match.get("_id", ""))
            }
        return {"found": False}
    except Exception as e:
        print(f"Error checking similar questions: {e}")
        return {"found": False}

def detect_followup_intent(current_question: str, last_question: str = "", last_answer: str = "", chat_id: str = None) -> bool:
    """Enhanced follow-up detection with better context awareness"""
    if not current_question or not last_question or not last_answer:
        return False
    
    # First try OpenAI detection
    try:
        prompt = f"""
You are an expert conversation analyst. Analyze whether the current question is a follow-up to the previous question and answer within the same conversation.

Previous Question: "{last_question}"
Previous Answer: "{last_answer}"
Current Question: "{current_question}"

A follow-up question:
1. References or builds upon the previous answer
2. Asks for clarification, elaboration, or more details about the previous answer
3. Uses pronouns or references (it, this, that, those, they, etc.) referring to previous content
4. Continues the same topic or asks for related information
5. Makes sense only in context of the previous Q&A pair
6. Asks for examples, more details, or expansion on previous answer
7. Uses phrases like "tell me more", "what about", "how about", "also", "additionally"

Important: A question is NOT a follow-up if:
- It's a completely new topic unrelated to the previous answer
- It doesn't reference the previous answer in any way
- It's an independent question that makes sense on its own

Respond with only "YES" if it's clearly a follow-up, or "NO" if it's independent.
"""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Answer only with YES or NO. Be conservative - only say YES if it's clearly a follow-up."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=10,
            temperature=0.1
        )
        result = response.choices[0].message.content.strip().upper()
        return result == "YES"
    except Exception as e:
        print(f"Error in OpenAI follow-up detection: {e}")
        return _enhanced_fallback_followup_detection(current_question, last_question, last_answer)

def _enhanced_fallback_followup_detection(current_question: str, last_question: str = "", last_answer: str = "") -> bool:
    """Enhanced fallback keyword-based follow-up detection"""
    if not current_question or not last_question or not last_answer:
        return False
    
    current_lower = current_question.lower().strip()
    last_answer_lower = last_answer.lower()
    
    # Strong follow-up indicators
    strong_indicators = [
        "tell me more", "more about", "elaborate", "explain further", "can you explain",
        "what about", "how about", "also", "additionally", "furthermore", "what else",
        "other", "more details", "expand on", "continue", "go on", "what do you mean",
        "clarify", "specify", "you said", "you mentioned", "from that", "based on that",
        "according to", "regarding", "concerning", "in relation to", "with respect to"
    ]
    
    for indicator in strong_indicators:
        if indicator in current_lower:
            return True
    
    # Check for pronouns that likely refer to previous content
    pronouns = ["it", "this", "that", "these", "those", "they", "them", "which", "such"]
    for pronoun in pronouns:
        # Check if pronoun appears at start or after common question words
        patterns = [
            f"^{pronoun}\\s",  # At the beginning
            f"\\b(what|how|why|when|where)\\s+{pronoun}\\s",  # After question words
            f"\\s{pronoun}\\s"  # In the middle
        ]
        for pattern in patterns:
            if re.search(pattern, current_lower):
                return True
    
    # Check for direct references to content that appeared in the last answer
    # Extract key terms from the last answer
    answer_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', last_answer_lower))
    question_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', current_lower))
    
    # If current question contains specific terms from the last answer, it might be a follow-up
    common_terms = answer_words.intersection(question_words)
    if len(common_terms) >= 2:  # At least 2 common significant terms
        return True
    
    # Check for short questions that are likely follow-ups
    if len(current_question.split()) <= 5:
        follow_up_starters = ["more", "else", "other", "how", "why", "what", "when", "where", "which"]
        if any(current_lower.startswith(word) for word in follow_up_starters):
            return True
    
    return False

def get_recent_context(chat_id: str, limit: int = 3):
    """Get recent Q&A pairs for a specific chat_id with better error handling"""
    try:
        recent_chats = list(collection.find(
            {"chat_id": chat_id}
        ).sort("timestamp", -1).limit(limit))
        
        # Return in chronological order (oldest first)
        return recent_chats[::-1] if recent_chats else []
    except Exception as e:
        print(f"Error getting recent context: {e}")
        return []