import json
import os
from dotenv import load_dotenv
from evaluator.rag_eval_pipeline import RAG_auto_Eval_pipeline

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

file_pairs = [
    ("data/sample-chat-conversation-01.json", "data/sample_context_vectors-01.json"),
    ("data/sample-chat-conversation-02.json", "data/sample_context_vectors-02.json"),
]
evaluator = RAG_auto_Eval_pipeline(api_key=API_KEY)
for chat_file, vector_file in file_pairs:
    with open(chat_file, "r") as f:
        chat_json = json.load(f)

    with open(vector_file, "r") as f:
        vector_json = json.load(f)

    individual_turn_reports, overall_summary = evaluator.conversation_pipeline(
        chat_json, vector_json
    )

    print("\n" + "=" * 70)
    print(f"FINAL CONVERSATION EVALUATION REPORT → {chat_file}")
    print("=" * 70)

    print("\n1. Individual Turn Pair Evaluations")
    print(json.dumps(individual_turn_reports, indent=4))

    print("\n2. Overall Conversation Summary")
    print(json.dumps(overall_summary, indent=4))

    # Save results to JSON file
    result_filename = f"results_{chat_file.split('/')[-1]}"
    with open(result_filename, "w") as f:
        json.dump({
            "individual_turn_reports": individual_turn_reports,
            "overall_summary": overall_summary
        }, f, indent=4)
    print(f"\nResults saved to {result_filename}")
