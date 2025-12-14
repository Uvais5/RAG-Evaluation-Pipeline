# RAG Evaluation Pipeline

This project implements an automated evaluation pipeline for RAG (Retrieval-Augmented Generation) systems. It evaluates chat conversations between users and AI assistants, assessing relevance, completeness, hallucination rates, and other metrics to determine if responses pass, fail, or need review.


## Architecture of the Evaluation Pipeline

The evaluation pipeline is centered around the `RAG_auto_Eval_pipeline` class in `evaluator/rag_eval_pipeline.py`. Here's the high-level architecture:

- **Initialization:** Loads the SentenceTransformer model for embeddings and initializes the Gemini client for LLM-based judgments.
- **Conversation Pipeline:** Processes entire conversations, extracting user-AI turn pairs.
- **Turn Evaluation:** For each turn pair:
  - Computes embedding-based relevance using cosine similarity between user query and AI response.
  - Uses LLM-as-a-judge (Gemini) to score relevance and completeness.
  - Blends embedding and LLM scores for final relevance.
  - Identifies potential claims in the AI response and checks support against provided context vectors using embeddings.
  - Calculates hallucination rate based on unsupported claims.
  - Estimates cost based on token count.
  - Applies thresholds to determine verdict (PASS, REVIEW, FAIL).
- **Aggregation:** Computes average metrics and verdict summaries across all turns in a conversation.

Key components:
- Embedding model: `all-MiniLM-L12-v2` for efficient text embeddings.
- LLM Judge: Gemini model for nuanced scoring.
- Similarity thresholds: Configurable for relevance and hallucination detection.



## Design Decisions

This architecture was chosen for several reasons:

- **Hybrid Evaluation:** Combining embedding-based similarity (fast, scalable) with LLM judgment (accurate, contextual) provides robust evaluation without relying solely on one method.
- **Modular Design:** The pipeline is broken into methods for cosine similarity, claim detection, embedding computation, and LLM calls, allowing easy testing and modification.
- **Threshold-Based Verdicts:** Using configurable thresholds enables fine-tuning for different use cases while providing clear pass/fail criteria.
- **Cost Estimation:** Including token-based cost calculation helps monitor expenses in production.
- **Error Handling:** Fallback scores prevent pipeline failures due to API issues.

Alternatives considered:
- Pure LLM evaluation: More accurate but slower and more expensive.
- Rule-based systems: Faster but less adaptable to complex scenarios.
- This hybrid approach balances speed, accuracy, and cost.

## Scalability for Millions of Daily Conversations

To ensure low latency and minimal costs at scale:

- **Latency Optimizations:**
  - Use efficient embedding models like `all-MiniLM-L12-v2` (fast inference).
  - Batch embedding computations where possible.
  - Cache frequently used embeddings to avoid recomputation.
  - Parallelize turn evaluations using async processing or multiprocessing.
  - Optimize LLM calls by using smaller, faster models (e.g., switch from `gemini-1.5-pro` to `gemini-1.5-flash`).

- **Cost Minimizations:**
  - Select cost-effective LLM models and monitor token usage.
  - Implement sampling for large datasets during development/testing.
  - Use embedding-based pre-filtering to reduce LLM calls (e.g., only call LLM for borderline cases).
  - Set up usage monitoring and alerts to detect cost spikes.
  - Consider on-premise models or cheaper alternatives for high-volume scenarios.

- **Infrastructure Considerations:**
  - Deploy on scalable cloud platforms (e.g., AWS Lambda, Google Cloud Functions) for auto-scaling.
  - Use message queues (e.g., SQS, Pub/Sub) for asynchronous processing.
  - Implement caching layers (Redis) for embeddings and results.
  - Database storage for historical evaluations with indexing for fast queries.


This design ensures the pipeline can handle high volumes while maintaining real-time performance and controlling costs.

## Local Setup Instructions

1. **Clone or navigate to the repository:**
   ```
   git clone https://github.com/Uvais5/RAG-Evaluation-Pipeline.git
   cd "RAG-Evaluation-Pipeline"
   ```

2. **Create a virtual environment:**
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   - Create a `.env` file in the root directory.
   - Add your Gemini API key:
     ```
     GEMINI_API_KEY=your_actual_api_key_here
     ```
   - Ensure `.env` is added to `.gitignore` to avoid committing sensitive data.

5. **Prepare data files:**
   - Place chat conversation JSON files in the `data/` directory (e.g., `sample-chat-conversation-01.json`).
   - Place corresponding context vector JSON files in the `data/` directory (e.g., `sample_context_vectors-01.json`).

6. **Run the evaluation:**
   ```
   python main.py
   ```
   - Results will be printed to the console and saved as JSON files (e.g., `results_sample-chat-conversation-01.json`).
  - The generated result JSONs are included in the repository for reference.





