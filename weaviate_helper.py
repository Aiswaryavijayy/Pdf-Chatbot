import os
import re
import weaviate
import weaviate.classes as wvc
from weaviate.auth import AuthApiKey
from weaviate.exceptions import WeaviateBaseError
from dotenv import load_dotenv
from openai import OpenAI
import logging
from typing import List, Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# OpenAI Client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_openai_embedding(text: str):
    """Returns a 1536-dim embedding for text"""
    try:
        # Clean and preprocess text
        text = re.sub(r'\s+', ' ', text.strip())
        resp = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text[:8192]  # Truncate to avoid OpenAI input limits
        )
        return resp.data[0].embedding
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        return None

# Weaviate Client
def get_weaviate_client():
    """Initialize and return Weaviate client with error handling"""
    try:
        client = weaviate.connect_to_weaviate_cloud(
            cluster_url=os.getenv("WEAVIATE_URL"),
            auth_credentials=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
        logger.info("Successfully connected to Weaviate")
        return client
    except WeaviateBaseError as e:
        logger.error(f"Failed to connect to Weaviate: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error connecting to Weaviate: {e}")
        raise

def create_schemas(chat_id: str):
    """Create Weaviate schema with chat_id-specific class"""
    class_name = f"DocumentQA_{chat_id.replace('-', '_')}"
    client = None
    try:
        client = get_weaviate_client()
        # Delete existing class if present
        if client.collections.exists(class_name):
            client.collections.delete(class_name)
            logger.info(f"Deleted existing class: {class_name}")
        # Create new class
        client.collections.create(
            name=class_name,
            description=f"PDF pages for chat {chat_id} with OpenAI embeddings",
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            properties=[
                wvc.config.Property(
                    name="chunk",
                    data_type=wvc.config.DataType.TEXT,
                    description="Page content of the PDF",
                    vectorize_property=False
                ),
                wvc.config.Property(
                    name="page_num",
                    data_type=wvc.config.DataType.INT,
                    description="Page number",
                    vectorize_property=False
                ),
                wvc.config.Property(
                    name="chunk_title",
                    data_type=wvc.config.DataType.TEXT,
                    description="Extracted title or heading from chunk",
                    vectorize_property=False
                ),
                wvc.config.Property(
                    name="chunk_keywords",
                    data_type=wvc.config.DataType.TEXT,
                    description="Extracted keywords from chunk",
                    vectorize_property=False
                )
            ],
            vector_index_config=wvc.config.Configure.VectorIndex.hnsw(
                distance_metric=wvc.config.VectorDistances.COSINE,
                ef_construction=200,  # Increased for better recall
                max_connections=64   # Increased for better connectivity
            )
        )
        logger.info(f"Created schema for class: {class_name}")
    except WeaviateBaseError as e:
        logger.error(f"Error creating schema for {class_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating schema: {e}")
        raise
    finally:
        if client:
            client.close()

def extract_keywords_and_title(text: str) -> Dict[str, str]:
    """Extract keywords and potential title from text chunk"""
    lines = text.split('\n')
    
    # Extract potential title (first non-empty line or lines in caps/title case)
    title = ""
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        if line and (line.isupper() or line.istitle() or len(line.split()) <= 8):
            title = line
            break
    
    # Extract keywords using simple heuristics
    words = re.findall(r'\b[A-Za-z]{4,}\b', text.lower())
    # Remove common stop words
    stop_words = {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'more', 'very', 'what', 'know', 'just', 'first', 'into', 'over', 'think', 'also', 'your', 'work', 'life', 'only', 'can', 'still', 'should', 'after', 'being', 'now', 'made', 'before', 'here', 'through', 'when', 'where', 'much', 'some', 'these', 'many', 'then', 'them', 'well', 'were'}
    keywords = [word for word in set(words) if word not in stop_words and len(word) > 3]
    
    return {
        'title': title,
        'keywords': ' '.join(keywords[:20])  # Top 20 keywords
    }

def answer_question_with_llm(question: str, context: str, is_followup: bool = False) -> str:
    """Answer question using OpenAI based on context with improved prompting"""
    if is_followup:
        system_prompt = """You are an AI assistant answering questions about a PDF document.
This is a follow-up question that may reference previous conversation context.
Use the provided context from the PDF (including page numbers) to answer accurately.
If the answer is not clearly stated in the context, say 'The document does not contain specific information about this question.'
Be conversational, comprehensive, and always reference page numbers when relevant.
If the question asks for comparisons, lists, or detailed explanations, provide them based on the context."""
    else:
        system_prompt = """You are an AI assistant answering questions about a PDF document.
Use the provided context from the PDF (including page numbers) to answer the user's question accurately and comprehensively.
If the answer is not clearly stated in the context, say 'The document does not contain specific information about this question.'
Be detailed, accurate, and always reference page numbers when applicable.
If the question asks for lists, comparisons, or explanations, provide comprehensive answers based on the available context."""
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4",  # Upgraded to GPT-4 for better reasoning
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context from PDF:\n{context}\n\nQuestion: {question}"}
            ],
            temperature=0.1,  # Lower temperature for more consistent answers
            max_tokens=800    # Increased for more detailed responses
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"Error in OpenAI completion: {e}")
        return "Error processing question with AI."

def get_answer_from_pdf(question: str, k: int = 5, chat_id: str = None, is_followup: bool = False) -> str:
    """Answer question using top-k pages with improved retrieval and relevance scoring"""
    if not chat_id:
        logger.error("No chat_id provided for answer retrieval")
        return "Error: No chat session specified."
    
    # Try multiple search strategies
    results = []
    
    # Strategy 1: Direct semantic search
    semantic_results = search_similar_question(question, k, chat_id)
    results.extend(semantic_results)
    
    # Strategy 2: Keyword-based search for specific terms
    keyword_results = search_with_keywords(question, k//2, chat_id)
    results.extend(keyword_results)
    
    # Remove duplicates and sort by distance
    seen_pages = set()
    unique_results = []
    for r in results:
        page_key = (r['page_num'], r['chunk'][:100])  # Use page and chunk start as key
        if page_key not in seen_pages:
            seen_pages.add(page_key)
            unique_results.append(r)
    
    # Sort by distance (most relevant first)
    unique_results.sort(key=lambda x: x['distance'])
    results = unique_results[:k]  # Take top k results
    
    if not results:
        logger.warning(f"No relevant pages found for question: {question}")
        return "No relevant information found in the PDF."
    
    # Improved relevance scoring
    distances = [r['distance'] for r in results]
    min_distance = min(distances)
    avg_distance = sum(distances) / len(distances)
    
    logger.info(f"Distances for question '{question}': {distances}")
    logger.info(f"Min distance: {min_distance}, Avg distance: {avg_distance}")
    
    # Dynamic threshold based on question complexity and results quality
    base_threshold = 0.5
    if len(question.split()) > 10:  # Complex questions get higher threshold
        threshold = base_threshold + 0.1
    elif any(word in question.lower() for word in ['what', 'how', 'why', 'when', 'where', 'who']):
        threshold = base_threshold + 0.05  # Question words get slightly higher threshold
    else:
        threshold = base_threshold
    
    # If we have multiple good results, lower the threshold
    if len([d for d in distances if d < 0.4]) >= 2:
        threshold = 0.6
    
    if min_distance > threshold:
        logger.info(f"Question '{question}' deemed unrelated (min distance: {min_distance}, threshold: {threshold})")
        return "This question does not seem to be related to the uploaded PDF."
    
    # Build comprehensive context
    context_parts = []
    for i, r in enumerate(results):
        context_parts.append(f"[Page {r['page_num']}]: {r['chunk']}")
    
    context = "\n\n".join(context_parts)
    logger.info(f"Retrieved context length: {len(context)} characters from {len(results)} pages")
    
    return answer_question_with_llm(question, context, is_followup=is_followup)

def search_with_keywords(question: str, k: int, chat_id: str) -> List[Dict]:
    """Search using keyword extraction and BM25-like scoring"""
    if not chat_id:
        return []
    
    class_name = f"DocumentQA_{chat_id.replace('-', '_')}"
    client = None
    
    try:
        client = get_weaviate_client()
        if not client.collections.exists(class_name):
            return []
        
        coll = client.collections.get(class_name)
        
        # Extract keywords from question
        question_keywords = re.findall(r'\b[A-Za-z]{3,}\b', question.lower())
        question_keywords = [kw for kw in question_keywords if kw not in {'the', 'and', 'are', 'you', 'for', 'not', 'can', 'how', 'what', 'why', 'when', 'where', 'who'}]
        
        if not question_keywords:
            return []
        
        # Use Weaviate's where filter with keyword matching
        where_filter = wvc.query.Filter.by_property("chunk").contains_any(question_keywords[:5])  # Top 5 keywords
        
        res = coll.query.fetch_objects(
            where=where_filter,
            limit=k,
            return_properties=["chunk", "page_num"]
        )
        
        # Calculate simple keyword-based relevance scores
        results = []
        for obj in res.objects:
            chunk_lower = obj.properties["chunk"].lower()
            score = sum(1 for kw in question_keywords if kw in chunk_lower)
            # Convert score to distance-like metric (lower is better)
            distance = max(0, 1.0 - (score / len(question_keywords)))
            
            results.append({
                "chunk": re.sub(r'\s+', ' ', obj.properties["chunk"].strip()),
                "page_num": obj.properties["page_num"],
                "distance": distance
            })
        
        return sorted(results, key=lambda x: x['distance'])[:k]
        
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return []
    finally:
        if client:
            client.close()

def store_pdf_chunks(pages, chat_id: str):
    """Store PDF pages with enhanced metadata in Weaviate"""
    if not chat_id:
        logger.error("No chat_id provided for storing chunks")
        raise ValueError("chat_id is required")
    
    class_name = f"DocumentQA_{chat_id.replace('-', '_')}"
    client = None
    
    try:
        client = get_weaviate_client()
        coll = client.collections.get(class_name)
        
        batch_size = 25  # Reduced batch size for better reliability
        successful_inserts = 0
        
        for i in range(0, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            
            for page in batch:
                chunk = page['text'].strip()
                if not chunk or len(chunk) < 50:  # Skip very short chunks
                    logger.warning(f"Skipping short/empty page {page['page_num']}")
                    continue
                
                # Extract metadata
                metadata = extract_keywords_and_title(chunk)
                
                # Generate embedding
                vector = get_openai_embedding(chunk)
                if not vector:
                    logger.warning(f"Skipping page {page['page_num']} due to embedding failure")
                    continue
                
                try:
                    coll.data.insert(
                        properties={
                            "chunk": chunk[:15000],  # Increased limit but still safe
                            "page_num": page['page_num'],
                            "chunk_title": metadata['title'][:500],  # Limit title length
                            "chunk_keywords": metadata['keywords'][:1000]  # Limit keywords length
                        },
                        vector=vector
                    )
                    successful_inserts += 1
                    logger.info(f"Stored page {page['page_num']} for chat {chat_id}")
                    
                except Exception as e:
                    logger.error(f"Error inserting page {page['page_num']}: {e}")
                    continue
        
        logger.info(f"Successfully stored {successful_inserts}/{len(pages)} pages for chat {chat_id}")
        
    except WeaviateBaseError as e:
        logger.error(f"Error storing chunks for {class_name}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error storing chunks: {e}")
        raise
    finally:
        if client:
            client.close()

def search_similar_question(question: str, k: int = 5, chat_id: str = None):
    """Return top-k pages similar to the question with improved search"""
    if not chat_id:
        logger.error("No chat_id provided for search")
        return []
    
    class_name = f"DocumentQA_{chat_id.replace('-', '_')}"
    client = None
    
    try:
        client = get_weaviate_client()
        if not client.collections.exists(class_name):
            logger.warning(f"Collection {class_name} does not exist")
            return []
        
        coll = client.collections.get(class_name)
        
        # Process and enhance the question
        processed_question = _process_and_enhance_question(question)
        
        q_vec = get_openai_embedding(processed_question)
        if not q_vec:
            logger.error(f"Failed to generate embedding for question: {question}")
            return []
        
        # Increase search limit to get more candidates
        search_limit = min(k * 3, 20)  # Search more, then filter
        
        res = coll.query.near_vector(
            near_vector=q_vec,
            limit=search_limit,
            return_metadata=wvc.query.MetadataQuery(distance=True, certainty=True),
            return_properties=["chunk", "page_num", "chunk_title", "chunk_keywords"]
        )
        
        results = []
        for obj in res.objects:
            # Clean up the chunk text
            chunk_text = re.sub(r'\s+', ' ', obj.properties["chunk"].strip())
            
            results.append({
                "chunk": chunk_text,
                "page_num": obj.properties["page_num"],
                "distance": obj.metadata.distance,
                "certainty": obj.metadata.certainty if hasattr(obj.metadata, 'certainty') else None,
                "chunk_title": obj.properties.get("chunk_title", ""),
                "chunk_keywords": obj.properties.get("chunk_keywords", "")
            })
        
        # Sort by distance and return top k
        results.sort(key=lambda x: x['distance'])
        final_results = results[:k]
        
        logger.info(f"Retrieved {len(final_results)} pages for question: {question}")
        return final_results
        
    except WeaviateBaseError as e:
        logger.error(f"Error in similarity search for {class_name}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in similarity search: {e}")
        return []
    finally:
        if client:
            client.close()

def _process_and_enhance_question(question: str) -> str:
    """Enhanced question processing with better follow-up handling"""
    # First handle follow-up markers
    processed = _process_followup_question(question)
    
    # Expand common abbreviations and synonyms
    expansions = {
        "what's": "what is",
        "how's": "how is",
        "can't": "cannot",
        "won't": "will not",
        "don't": "do not",
    }
    
    for abbrev, expansion in expansions.items():
        processed = processed.replace(abbrev, expansion)
    
    # Add context words for better embedding
    question_words = processed.lower().split()
    if any(word in question_words for word in ['define', 'definition', 'meaning', 'what is']):
        processed += " definition explanation meaning"
    elif any(word in question_words for word in ['how', 'process', 'steps']):
        processed += " process method procedure steps"
    elif any(word in question_words for word in ['why', 'reason', 'cause']):
        processed += " reason cause explanation rationale"
    elif any(word in question_words for word in ['list', 'examples', 'types']):
        processed += " list examples types categories"
    
    return processed

def _process_followup_question(question: str) -> str:
    """Improved follow-up question processing"""
    # Enhanced follow-up markers
    followup_patterns = [
        r'based on (?:our )?previous conversation:?\s*',
        r'previous q:?\s*',
        r'previous a:?\s*',
        r'earlier q:?\s*',
        r'earlier a:?\s*',
        r'follow-?up:?\s*',
        r'continuing from (?:above|before):?\s*',
        r'as mentioned (?:earlier|before):?\s*',
        r'regarding (?:the )?(?:above|previous):?\s*'
    ]
    
    question_lower = question.lower()
    
    for pattern in followup_patterns:
        match = re.search(pattern, question_lower)
        if match:
            # Extract the actual question part after the marker
            start_pos = match.end()
            actual_question = question[start_pos:].strip()
            
            # Remove any additional follow-up indicators
            actual_question = re.sub(r'^follow-up\s*\([^)]+\):\s*', '', actual_question, flags=re.IGNORECASE)
            
            if actual_question and len(actual_question) > 5:
                logger.info(f"Processed follow-up question from '{question}' to '{actual_question}'")
                return actual_question
    
    return question

def get_document_stats(chat_id: str) -> Dict:
    """Get statistics about the stored document"""
    if not chat_id:
        return {}
    
    class_name = f"DocumentQA_{chat_id.replace('-', '_')}"
    client = None
    
    try:
        client = get_weaviate_client()
        if not client.collections.exists(class_name):
            return {"error": "Document not found"}
        
        coll = client.collections.get(class_name)
        
        # Get total count
        result = coll.aggregate.over_all(total_count=True)
        total_chunks = result.total_count
        
        # Get page range
        res = coll.query.fetch_objects(
            limit=1000,  # Adjust based on your needs
            return_properties=["page_num"]
        )
        
        page_nums = [obj.properties["page_num"] for obj in res.objects]
        min_page = min(page_nums) if page_nums else 0
        max_page = max(page_nums) if page_nums else 0
        
        return {
            "total_chunks": total_chunks,
            "page_range": f"{min_page}-{max_page}",
            "total_pages": len(set(page_nums))
        }
        
    except Exception as e:
        logger.error(f"Error getting document stats: {e}")
        return {"error": str(e)}
    finally:
        if client:
            client.close()