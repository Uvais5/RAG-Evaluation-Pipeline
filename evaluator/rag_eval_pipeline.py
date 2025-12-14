import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

from sentence_transformers import SentenceTransformer
from google import genai
import re
embeding_model_name = "all-MiniLM-L12-v2"
llm_judge_model_name = "gemini-2.5-flash"

Similirity_rang = 0.55
Similirity_out_rang = 0.35

relavance_threshold = 0.65
completeness_threshold = 0.6
hallucination_threshold = 0.2

model_price_per_token = 0.50
class RAG_auto_Eval_pipeline:
  def __init__(self,api_key):
    self.embdding_model = SentenceTransformer(embeding_model_name)
    self.api_key = api_key
    try:
      self.llm_judge = genai.Client(api_key=self.api_key)
      print("Gemini is Load.")
    except Exception:
        print("There is problem in gemini.")
  def cosine_sim(self,a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    nor = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return nor
  def token_size(self, text: str) -> int:
      cal = int(len(text.split())*1.5)
      return cal
  def llm_as_judge(self, user, ai_chatbot):
    prompt = f"""
User Question:
{user}

AI Response:
{ai_chatbot}

Evaluate on:
- relevance_score (0.0–1.0)
- completeness_score (0.0–1.0)

Completeness means: fully answers the user's question without missing key details.

Return ONLY valid JSON:
{{"relevance_score": float, "completeness_score": float}}
"""

    try:
        resp = self.llm_judge.models.generate_content(
            model=llm_judge_model_name,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        raw = resp.text.strip()
        raw = raw.replace("```json", "").replace("```", "")
        parsed = json.loads(raw)

        return {
            "relevance_score": float(parsed.get("relevance_score", 0.5)),
            "completeness_score": float(parsed.get("completeness_score", 0.5))
        }

    except Exception:
        # Safe fallback instead of killing evaluation
        return {"relevance_score": 0.5, "completeness_score": 0.5}

  
  #claim Logic
  def is_potential_claim(self,text: str):
      
    sentences = [s.strip() for s in text.split(".") if s.strip()]
    claims = []

    for s in sentences:
        if (
            re.search(r"\d", s) or
            re.search(r"\b(is|are|was|were|will|has|have|includes|costs|provides)\b", s.lower()) or
            s[0].isupper()
        ):
            claims.append(s)

    return claims
  def get_embeding(self,text):
    # model = SentenceTransformer(embeding_model_name)
    return self.embdding_model.encode(text)

    
  def evaluation_turn(
      self, 
      user_side,
      ai_side,
      chat_id,
      vector_json):
    user_message = user_side["message"]
    ai_message = ai_side["message"]
    # print(user_message)
    # print("working_evaluation")
    try:
      t1 = datetime.fromisoformat(user_side["created_at"].replace("Z", "+00:00"))
      t2 = datetime.fromisoformat(ai_side["created_at"].replace("Z", "+00:00"))
      latency_ms = int((t2 - t1).total_seconds() * 1000)
    except Exception:
      latency_ms = None
    #embedding based relavance
    user_embd = self.get_embeding(user_message)
    ai_embd = self.get_embeding(ai_message)
    emd_matrix = self.cosine_sim(user_embd,ai_embd)
    # LLM as Judge
    judge_score = self.llm_as_judge(user_message,ai_message)
    #Final blended score
    relevance_score = round((emd_matrix + judge_score["relevance_score"]) / 2, 4)
    completeness_score = round(judge_score["completeness_score"], 4)
    context_docs = [d["text"] for d in vector_json.get("data", {}).get("vector_data", []) 
    if d.get("text")]
    
    context_emd = [self.get_embeding(c) for c in context_docs]
    claims =self.is_potential_claim(ai_message)
    # claims = self.is_potential_claim(ai_message)

    supported, unsupported = 0, 0

    if claims:
        for c in claims:
            claim_emb = self.get_embeding(c)
            max_sim = max(
                (self.cosine_sim(claim_emb, ctx) for ctx in context_emd),
                default=0.0
            )

            if max_sim >= Similirity_rang:
                supported += 1
            elif max_sim < Similirity_out_rang:
                unsupported += 1

        hallucination_rate = unsupported / len(claims)
    else:
        # No factual claims → no hallucination
        hallucination_rate = 0.0


    # hallucination_rate = unsupported /max(len(claims),1)
    hallucination_rate = (
        unsupported / len(claims)
        if len(claims) >= 2
        else 0.0
    )

    # cost calculation
    cost_call = (self.token_size(ai_message)/1000)*model_price_per_token
    #verdict
    if (
        relevance_score >= relavance_threshold and
        completeness_score >= completeness_threshold and
        hallucination_rate <= hallucination_threshold
    ):
        verdict = "PASS"
    elif hallucination_rate >= 0.6 or relevance_score < 0.4:
        verdict = "FAIL"
    else:
        verdict = "REVIEW"

    return {
            "chat_id": chat_id,
            "turn_pair": f"Turn {user_side['turn']} -> Turn {ai_side['turn']}",
            "latency_ms": latency_ms,
            "relevance_score": relevance_score,
            "completeness_score": completeness_score,
            "supported_claims": supported,
            "unsupported_claims": unsupported,
            "hallucination_rate": round(hallucination_rate, 4),
            "estimated_cost": round(cost_call, 6),
            "verdict": verdict,
            "user_message": user_message,
            "assistant_message": ai_message
        }
  def get_report(self, chat_turns, chat_id, 
                 chat_json, vector_json):
    report = []
    for i in range(len(chat_turns) - 1):
        user_message = chat_turns[i]
        ai_message = chat_turns[i + 1]
      
        is_user = user_message.get("sender_id") == chat_json.get("user_id")
        is_ai = ai_message.get("sender_id") != chat_json.get("user_id")
    
        if is_user and is_ai:
            # print(is_user)
            if user_message.get("message") and ai_message.get("message"):
                get_report = self.evaluation_turn(
                    user_message,
                    ai_message,
                    chat_id,
                    vector_json
                )
                report.append(get_report)
      
        
    if not report:
        return [], {"error": "No valid User → AI turn pairs found"}
    return report

  def conversation_pipeline(self,chat_json,vector_json):
    chat_turns = chat_json.get("conversation_turns", [])
    chat_id = chat_json.get("chat_id")

    report = self.get_report(chat_turns, chat_id, chat_json, vector_json)
    # report = []
    
   


    # Aggregate Conversation Metrics
    summary = {
        "chat_id": chat_id,
        "total_turn_pairs_evaluated": len(report),
        "metrics_averages": {
            "avg_relevance_score": round(
                np.mean([r["relevance_score"] for r in report]), 4
            ),
            "avg_completeness_score": round(
                np.mean([r["completeness_score"] for r in report]), 4
            ),
            "avg_latency_ms": round(
                np.mean([r["latency_ms"] for r in report if r["latency_ms"]]), 2
            ),
        },
        "verdict_summary": {
            "PASS": sum(1 for r in report if r["verdict"] == "PASS"),
            "REVIEW": sum(1 for r in report if r["verdict"] == "REVIEW"),
            "FAIL": sum(1 for r in report if r["verdict"] == "FAIL"),
        }
    }

    return report, summary

