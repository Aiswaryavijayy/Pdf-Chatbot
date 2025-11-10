import os
import re
import uuid
import random
import io
import datetime
import streamlit as st
import PyPDF2
from fpdf import FPDF
from dotenv import load_dotenv
from openai import OpenAI

from weaviate_helper import create_schemas, store_pdf_chunks, get_answer_from_pdf
from mongo_helper import save_chat, check_similar_question, detect_followup_intent, get_recent_context

# ——— Setup ———
load_dotenv()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not hasattr(st, "experimental_rerun"):
    st.experimental_rerun = lambda: None

# ——— Helpers ———

def generate_chat_id():
    return str(uuid.uuid4())

def clean_answer(ans: str) -> str:
    return re.sub(r'(?i)(\(page \d+\)|page \d+:|\[page \d+\]|source: page \d+|page \d+)', '', ans).strip()

def clean_text(text: str) -> str:
    return re.sub(r'(?i)(\(page \d+\)|page \d+:|\[page \d+\]|source: page \d+|page \d+)', '', text).strip()

def extract_meaningful_heading(question: str) -> str:
    """Extract meaningful content from question to create a structured heading"""
    question = question.strip().rstrip('?!.')
    
    # Remove common question words and patterns
    question_lower = question.lower()
    
    # Handle specific patterns
    patterns = [
        # "Create a project on X" -> "Project on X"
        (r'^(?:create|make|build|develop)\s+(?:a\s+)?(.+)', r'\1'),
        # "What is X" -> "X"
        (r'^what\s+(?:is|are|was|were)\s+(.+)', r'\1'),
        # "Tell me about X" -> "X"
        (r'^(?:tell\s+me\s+about|explain|describe)\s+(.+)', r'\1'),
        # "How to X" -> "X"
        (r'^how\s+to\s+(.+)', r'\1'),
        # "Why is X" -> "X"
        (r'^why\s+(?:is|are|was|were|do|does|did)\s+(.+)', r'\1'),
        # "Where is X" -> "X"
        (r'^where\s+(?:is|are|was|were)\s+(.+)', r'\1'),
        # "When was X" -> "X"
        (r'^when\s+(?:is|are|was|were|did|does)\s+(.+)', r'\1'),
        # "Can you X" -> "X"
        (r'^(?:can\s+you|could\s+you|would\s+you)\s+(.+)', r'\1'),
        # "Give me X" -> "X"
        (r'^(?:give\s+me|show\s+me|provide\s+me)\s+(.+)', r'\1'),
        # "I want to know about X" -> "X"
        (r'^i\s+(?:want\s+to\s+know\s+about|need\s+to\s+know\s+about|would\s+like\s+to\s+know\s+about)\s+(.+)', r'\1'),
    ]
    
    # Apply patterns
    result = question
    for pattern, replacement in patterns:
        match = re.match(pattern, question_lower)
        if match:
            # Preserve original case but use the extracted part
            start_pos = match.start(1)
            end_pos = match.end(1)
            result = question[start_pos:end_pos]
            break
    
    # Clean up common filler words at the beginning
    filler_words = ['the', 'a', 'an', 'some', 'any', 'this', 'that', 'these', 'those']
    words = result.split()
    
    # Remove filler words from the beginning
    while words and words[0].lower() in filler_words:
        words.pop(0)
    
    if words:
        result = ' '.join(words)
    
    # Capitalize properly
    result = result.strip()
    if result:
        # Capitalize first letter and proper nouns
        words = result.split()
        capitalized_words = []
        for i, word in enumerate(words):
            if i == 0:  # First word
                capitalized_words.append(word.capitalize())
            elif word.lower() in ['on', 'in', 'at', 'by', 'for', 'with', 'to', 'of', 'and', 'or', 'but']:
                # Keep articles and prepositions lowercase unless they're the first word
                capitalized_words.append(word.lower())
            elif len(word) > 3:  # Capitalize longer words
                capitalized_words.append(word.capitalize())
            else:
                capitalized_words.append(word.lower())
        
        result = ' '.join(capitalized_words)
    
    # If result is too short or empty, fall back to original question
    if len(result.strip()) < 3:
        result = question.capitalize()
    
    # Limit length for heading
    if len(result) > 80:
        result = result[:77] + "..."
    
    return result

def generate_varied_followup_question(previous_answer: str, question_count: int = 1) -> str:
    templates = [
        f"Tell me more details about: {previous_answer[:100]}...",
        f"Can you elaborate further on: {previous_answer[:100]}...",
        f"What additional information is available about: {previous_answer[:100]}...",
        f"Please provide more context regarding: {previous_answer[:100]}...",
        f"I'd like to understand better: {previous_answer[:100]}...",
        f"Can you expand on this topic: {previous_answer[:100]}...",
        f"What else should I know about: {previous_answer[:100]}...",
        f"Please dive deeper into: {previous_answer[:100]}..."
    ]
    return templates[question_count % len(templates)]

def paraphrase_answer(original_answer: str, user_question: str) -> str:
    paraphrase_prompt = f"""
Please paraphrase this answer while keeping all the important information intact and maintaining perfect grammar:
\"{original_answer}\"

The original question was: \"{user_question}\"

Requirements:
- Use different words and sentence structures
- Keep the same meaning and all factual content
- Ensure proper grammar and natural flow
- Make it sound conversational and clear
- Don't add any new information not in the original
"""
    try:
        resp = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": paraphrase_prompt}],
            temperature=0.6,
            max_tokens=400
        )
        return resp.choices[0].message.content.strip()
    except:
        return original_answer

def generate_chat_summary(history: list) -> str:
    if not history:
        return "New Chat"
    topics = []
    for entry in history:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", entry['question'].lower())
        topics += [w for w in words if w not in ('what','where','when','how','why','does','can','will','should','about')]
    unique = list(dict.fromkeys(topics))[:3]
    if unique:
        return " & ".join(unique).title()
    q = history[0]['question']
    return (q[:30] + "...") if len(q) > 30 else q

def compute_title(chat: dict, first_q: str = None) -> str:
    if chat.get('name'):
        return chat['name']
    if first_q:
        fq = first_q.lower().strip()
        greets = ['hello','hi','hey','good morning','good afternoon','good evening']
        if fq in greets or fq.rstrip('!.') in greets:
            return "Greeting"
        m = re.match(r"^what(?: is|'s)\s+(.+?)[\?\.!]?$", fq)
        if m:
            return f"Summary of {m.group(1).capitalize()}"
        cleaned = re.sub(r"\b(what|where|when|how|why|does|can|will|should|about|tell|me|explain)\b", "", fq)
        words = [w for w in cleaned.split() if len(w) > 3][:3]
        if words:
            return " & ".join(w.capitalize() for w in words)
        return (first_q[:30] + "...") if len(first_q) > 30 else first_q
    return generate_chat_summary(chat['history'])

def is_answer_from_pdf(ans: str) -> bool:
    fallbacks = ["not mentioned","not found","not available","cannot find","no information"]
    return not any(f in ans.lower() for f in fallbacks)

def convert_assistant_response_to_not_found(ans: str) -> str:
    if any(phrase in ans.lower() for phrase in ["how can i assist","i'm here to help","what can i do"]):
        return "This information is not found in the document."
    return ans

def clean_text_for_pdf(text):
    """Clean and encode text for PDF compatibility"""
    # Replace problematic characters
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    text = text.replace('—', '-').replace('–', '-')
    text = text.replace('…', '...')
    
    # Remove or replace other non-ASCII characters
    text = ''.join(char if ord(char) < 128 else ' ' for char in text)
    
    # Clean up extra whitespace
    text = ' '.join(text.split())
    
    return text

def create_enhanced_pdf(chat_history):
    """Create an enhanced PDF with better formatting and smart heading extraction"""
    buffer = io.BytesIO()
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    
    # Add first page
    pdf.add_page()
    
    # Process each Q&A pair
    for i, entry in enumerate(chat_history, start=1):
        question = clean_text_for_pdf(entry['question'])
        answer = clean_text_for_pdf(clean_answer(entry['answer']))
        
        # Check if we need a new page (except for first entry)
        if i > 1 and pdf.get_y() > 240:
            pdf.add_page()
        
        # Extract meaningful heading from the question
        heading = extract_meaningful_heading(question)
        heading = clean_text_for_pdf(heading)
        
        # Section Header (centered)
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 12, heading, ln=True, align="C")
        pdf.ln(5)
        
        # Add underline
        pdf.set_draw_color(100, 100, 100)
        pdf.line(30, pdf.get_y(), 180, pdf.get_y())
        pdf.ln(10)
        
        # Answer Content
        pdf.set_font("Arial", "", 11)
        pdf.set_left_margin(25)
        
        if answer:
            # Calculate available width for text (page width - margins)
            page_width = pdf.w - 2 * pdf.l_margin - 5
            char_width = 2.5  # Approximate character width in mm for Arial 11pt
            max_chars_per_line = int(page_width / char_width)
            
            # Handle structured content
            if '\n' in answer:
                # Split into lines and handle each
                lines = [line.strip() for line in answer.split('\n') if line.strip()]
                
                for line_idx, line in enumerate(lines):
                    # Check if it looks like a bullet point or list item
                    if line.startswith(('-', '•', '*')) or any(line.startswith(f'{i}.') for i in range(1, 20)):
                        # Format as bullet point on separate line
                        pdf.set_font("Arial", "", 10)
                        # Remove existing bullet/number and add consistent bullet
                        clean_line = line.lstrip('-•*123456789. ').strip()
                        
                        # Handle long bullet points by splitting them
                        if len(clean_line) > max_chars_per_line:
                            # Split long bullet into multiple lines using word wrapping
                            words = clean_line.split()
                            current_line = ""
                            first_line = True
                            
                            for word in words:
                                # Check if adding this word would exceed the line
                                test_line = current_line + (" " + word if current_line else word)
                                if len(test_line) <= max_chars_per_line:
                                    current_line = test_line
                                else:
                                    # Print current line
                                    if current_line:
                                        if first_line:
                                            pdf.cell(10, 6, "-", ln=False)
                                            pdf.cell(0, 6, current_line, ln=True)
                                            first_line = False
                                        else:
                                            pdf.cell(10, 6, " ", ln=False)  # Indent continuation
                                            pdf.cell(0, 6, current_line, ln=True)
                                    current_line = word
                            
                            # Print remaining text
                            if current_line:
                                if first_line:
                                    pdf.cell(10, 6, "-", ln=False)
                                else:
                                    pdf.cell(10, 6, " ", ln=False)  # Indent continuation
                                pdf.cell(0, 6, current_line, ln=True)
                        else:
                            # Short bullet point - single line
                            pdf.cell(10, 6, "-", ln=False)
                            pdf.cell(0, 6, clean_line, ln=True)
                        
                        pdf.ln(2)  # Small space after each bullet
                        
                    elif line.startswith(('Key', 'Important', 'Note:', 'Summary', 'Overview')):
                        # Format as highlighted section header
                        pdf.ln(3)
                        pdf.set_font("Arial", "B", 11)
                        pdf.multi_cell(0, 6, line)
                        pdf.set_font("Arial", "", 11)
                        pdf.ln(2)
                    elif len(line) < 100 and line.endswith(':'):
                        # Likely a subheading
                        pdf.ln(2)
                        pdf.set_font("Arial", "B", 10)
                        pdf.multi_cell(0, 6, line)
                        pdf.set_font("Arial", "", 10)
                        pdf.ln(1)
                    else:
                        # Regular paragraph
                        pdf.set_font("Arial", "", 11)
                        pdf.multi_cell(0, 6, line)
                        if line_idx < len(lines) - 1:
                            pdf.ln(3)  # Space between paragraphs
            else:
                # Single paragraph answer - check if it contains inline bullet points
                if any(marker in answer for marker in [' - ', ' • ', ' * ']) or any(f'{i}. ' in answer for i in range(1, 10)):
                    # Split by common bullet patterns and format each as separate bullet
                    import re
                    # Split by bullet patterns while preserving the content
                    parts = re.split(r'(?:\s*[-•*]\s*|\d+\.\s*)', answer)
                    parts = [part.strip() for part in parts if part.strip()]
                    
                    for part_idx, part in enumerate(parts):
                        if part and len(part) > 3:  # Avoid empty or very short parts
                            # Handle long bullet points
                            if len(part) > max_chars_per_line:
                                words = part.split()
                                current_line = ""
                                first_line = True
                                
                                for word in words:
                                    test_line = current_line + (" " + word if current_line else word)
                                    if len(test_line) <= max_chars_per_line:
                                        current_line = test_line
                                    else:
                                        if current_line:
                                            if first_line:
                                                pdf.cell(10, 6, "-", ln=False)
                                                pdf.cell(0, 6, current_line, ln=True)
                                                first_line = False
                                            else:
                                                pdf.cell(10, 6, " ", ln=False)  # Indent
                                                pdf.cell(0, 6, current_line, ln=True)
                                        current_line = word
                                
                                if current_line:
                                    if first_line:
                                        pdf.cell(10, 6, "-", ln=False)
                                    else:
                                        pdf.cell(10, 6, " ", ln=False)
                                    pdf.cell(0, 6, current_line, ln=True)
                            else:
                                pdf.cell(10, 6, "-", ln=False)
                                pdf.cell(0, 6, part, ln=True)
                            
                            pdf.ln(2)  # Space after each bullet
                else:
                    # Regular single paragraph
                    pdf.multi_cell(0, 6, answer)
        else:
            pdf.set_font("Arial", "I", 10)
            pdf.multi_cell(0, 6, "No content available.")
        
        # Reset margins and add separator
        pdf.set_left_margin(20)
        pdf.ln(15)
        
        # Add separator line between sections (if not last item)
        if i < len(chat_history):
            pdf.set_draw_color(180, 180, 180)
            pdf.line(40, pdf.get_y(), 170, pdf.get_y())
            pdf.ln(20)
    
    # Save to buffer
    pdf.output(buffer)
    buffer.seek(0)
    
    return buffer

# ——— Streamlit Config ———
st.set_page_config(page_title="📄 PDF Q&A", layout="wide")

# ——— State Init ———
if 'chats' not in st.session_state:
    st.session_state.chats = {}
    st.session_state.current_chat_id = None
if 'yes_count' not in st.session_state:
    st.session_state.yes_count = {}

# ——— Sidebar ———
with st.sidebar:
    st.title("📑 Chats")
    if st.button("🆕 New Chat"):
        cid = generate_chat_id()
        st.session_state.chats[cid] = {"name": None, "history": [], "pdf_processed": False, "chat_id": cid}
        st.session_state.current_chat_id = cid
        st.session_state.yes_count[cid] = 0
        st.experimental_rerun()
    if st.session_state.chats:
        ids = list(st.session_state.chats)
        idx = ids.index(st.session_state.current_chat_id) if st.session_state.current_chat_id in ids else 0
        sel = st.selectbox(
            "Your Chats", ids, index=idx,
            format_func=lambda cid: compute_title(
                st.session_state.chats[cid],
                st.session_state.chats[cid]['history'][0]['question'] if st.session_state.chats[cid]['history'] else None
            )
        )
        st.session_state.current_chat_id = sel
    else:
        st.info("Start a new chat!")
    if st.button("🔄 Recreate Weaviate Schema"):
        create_schemas(st.session_state.current_chat_id)
        st.success("✅ Schema recreated.")

if st.session_state.current_chat_id is None:
    st.warning("Click 'New Chat' to begin.")
    st.stop()

chat = st.session_state.chats[st.session_state.current_chat_id]

# ——— Header & PDF Processing ———
first_q = chat['history'][0]['question'] if chat['history'] else None
st.title(f"📄 PDF Q&A - {compute_title(chat, first_q)}")
st.markdown("Upload a PDF, then ask questions using the chat bar below.")

uploader = st.file_uploader("Upload a PDF", type=["pdf"], key=f"up_{chat['chat_id']}")
if uploader and st.button("Process PDF", key=f"proc_{chat['chat_id']}"):
    reader = PyPDF2.PdfReader(uploader)
    pages = [{"text": clean_text(p.extract_text() or ""), "page_num": i+1} for i,p in enumerate(reader.pages)]
    st.info(f"Extracted {len(pages)} pages. Indexing…")
    create_schemas(chat['chat_id'])
    store_pdf_chunks(pages, chat['chat_id'])
    chat['pdf_processed'] = True
    st.success("✅ Indexed successfully!")

st.markdown("---")

# ——— Display Chat History ———
for entry in chat['history']:
    st.chat_message("user").write(entry['question'])
    resp = clean_answer(entry['answer'])
    if entry.get('suggestion'):
        resp += "\n\n" + entry['suggestion']
    st.chat_message("assistant").write(resp)

# ——— Chat Input ———
user_q = st.chat_input("Ask a question or type 'download pdf' to save chat...")
if user_q:
    st.chat_message("user").write(user_q)
    norm = user_q.lower().strip()

    # ——— Enhanced PDF Download Handler ———
    if any(k in norm for k in ("download","dwld","as a pdf","pdf format","give it")):
        if not chat['history']:
            st.warning("No chat history to download!")
        else:
            try:
                # Create enhanced PDF
                buffer = create_enhanced_pdf(chat['history'])
                
                # Generate filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"chat_history_{timestamp}.pdf"
                
                st.download_button(
                    "📥 Download Enhanced Chat PDF",
                    data=buffer,
                    file_name=filename,
                    mime="application/pdf",
                    key=f"dl_{chat['chat_id']}",
                    help="Download your complete chat history as a formatted PDF"
                )
                
                success_msg = f"✅ Enhanced PDF ready for download! Contains {len(chat['history'])} Q&A pairs with improved formatting."
                st.success(success_msg)
                st.chat_message("assistant").write(success_msg)
                
                chat['history'].append({"question": user_q, "answer": success_msg, "suggestion": None})
                save_chat(user_q, success_msg, is_followup=False, parent_question=None, chat_id=chat['chat_id'])
                
            except Exception as e:
                error_msg = f"❌ Error creating PDF: {str(e)}"
                st.error(error_msg)
                st.chat_message("assistant").write(error_msg)
                chat['history'].append({"question": user_q, "answer": error_msg, "suggestion": None})
                save_chat(user_q, error_msg, is_followup=False, parent_question=None, chat_id=chat['chat_id'])
        
        st.stop()

    # ——— Original Q&A Flow ———
    last = chat['history'][-1] if chat['history'] else {}
    negs = {"no","nope","nah","not now","no thanks","no thank you"}
    if norm not in {"yes","y","yep","sure","ok","okay"} and norm not in negs:
        st.session_state.yes_count[chat['chat_id']] = 0

    if norm in negs and last.get('suggestion'):
        r = "No problem! Let me know if you'd like to explore anything else."
        st.chat_message("assistant").write(r)
        chat['history'].append({"question": user_q, "answer": r, "suggestion": None})
        save_chat(user_q, r, is_followup=False, parent_question=last.get('question'), chat_id=chat['chat_id'])
        st.session_state.yes_count[chat['chat_id']] = 0
        st.rerun()
    elif norm in negs:
        r = "Please upload and process a PDF first." if not chat['pdf_processed'] else "I understand. Feel free to ask any questions."
        st.chat_message("assistant").write(r)
        chat['history'].append({"question": user_q, "answer": r, "suggestion": None})
        save_chat(user_q, r, is_followup=False, parent_question=last.get('question'), chat_id=chat['chat_id'])
        st.session_state.yes_count[chat['chat_id']] = 0
        st.rerun()

    # Determine follow‑up vs new query
    if norm in {"yes","y","yep","sure","ok","okay"}:
        is_fu = True
        cnt = st.session_state.yes_count[chat['chat_id']]
        follow_q = generate_varied_followup_question(last.get('answer',''), cnt)
        st.session_state.yes_count[chat['chat_id']] += 1
    else:
        is_fu = detect_followup_intent(user_q, last.get('question'), last.get('answer'), chat['chat_id'])
        follow_q = user_q

    sim = check_similar_question(follow_q, chat['chat_id'])
    if sim.get('found') and not is_fu:
        raw = sim['answer']
    elif not chat['pdf_processed']:
        raw = "Please upload and process a PDF first."
    else:
        if is_fu and last.get('question'):
            ctxs = get_recent_context(chat['chat_id'], limit=3)[-2:]
            ctx_text = "\n\n".join(f"Previous Q: {c['question']}\nPrevious A: {clean_answer(c['answer'])}" for c in ctxs)
            raw = get_answer_from_pdf(f"Context:\n{ctx_text}\n\nCurrent: {follow_q}", k=5, chat_id=chat['chat_id'], is_followup=True)
        else:
            raw = get_answer_from_pdf(follow_q, k=3, chat_id=chat['chat_id'], is_followup=False)

    answer = convert_assistant_response_to_not_found(clean_answer(raw))
    if norm in {"yes","y","yep","sure","ok","okay"} and st.session_state.yes_count[chat['chat_id']] > 1 and is_answer_from_pdf(answer):
        answer = paraphrase_answer(answer, follow_q)

    st.chat_message("assistant").write(answer)

    suggestion = None
    if chat['pdf_processed'] and answer and is_answer_from_pdf(answer) and norm not in negs:
        inv = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":
                f"You just answered: '{user_q}'. The answer: '{answer[:200]}...'. Generate ONE concise yes/no invitation to explore more on this PDF topic."
            }],
            temperature=0.9,
            max_tokens=30
        )
        suggestion = inv.choices[0].message.content.strip().strip('"')
        st.chat_message("assistant").write(suggestion)

    chat['history'].append({"question": user_q, "answer": answer, "suggestion": suggestion})
    save_chat(user_q, answer, is_followup=is_fu, parent_question=last.get('question'), chat_id=chat['chat_id'])
    st.rerun()