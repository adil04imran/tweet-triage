import pandas as pd
import os
import sys
import json
import time
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

from sklearn.metrics import classification_report, accuracy_score
from groq import Groq
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.agent.support_agent import AppleSupportAgent

# Groq Free Tier: ~30 req/min. Each ticket = 3 LLM calls + 1 judge call = 4 calls.
# 30 req/min / 4 calls = ~7.5 tickets/min => sleep 8s between tickets to be safe.
RATE_LIMIT_SLEEP_SECONDS = 8


def llm_judge(client, drafted_reply, historical_reply):
    """
    LLM-as-a-judge rubric for reply quality.
    Scores from 1 to 5.
    1 = Completely useless / hallucinates badly.
    3 = Okay, but too generic or misses nuance.
    5 = Excellent, grounded, matches or exceeds historical reply quality.
    """
    prompt = f"""You are an expert customer service quality evaluator.
Compare the AI-drafted reply to the historical human reply for a customer query.

Historical Reply (Ground Truth): "{historical_reply}"
AI-Drafted Reply: "{drafted_reply}"

Score the AI-Drafted Reply from 1 to 5 based on these 3 criteria:
- Tone: Is it empathetic, calm, and professional?
- Actionability: Does it provide a clear next step?
- Groundedness: Does it make sense given the context (no hallucinations)?

Output ONLY a single integer between 1 and 5. Do not explain."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="qwen/qwen3.8-27b",
            temperature=0.0
        )
        score_text = chat_completion.choices[0].message.content.strip()
        score = int(''.join(filter(str.isdigit, score_text[:3])))
        return min(max(score, 1), 5)
    except Exception:
        return 3  # fallback to neutral average if LLM fails


def run_evaluation(golden_csv_path, results_output_path):
    print("Loading Golden Set...")
    df = pd.read_csv(golden_csv_path)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not found. Make sure your .env file is set up correctly.")
    judge_client = Groq(api_key=api_key)

    agent = AppleSupportAgent(
        intents_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'discovered_intents.json'),
        db_path=os.path.join(os.path.dirname(__file__), '..', 'data', 'chroma_db')
    )

    results = []
    print(f"Evaluating Agent Pipeline on {len(df)} examples...")
    print(f"Rate-limiting to stay within Groq free tier (sleep={RATE_LIMIT_SLEEP_SECONDS}s between tweets)...")
    print(f"Estimated time: ~{len(df) * RATE_LIMIT_SLEEP_SECONDS // 60} minutes\n")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        tweet = row['text']
        true_intent = row['true_intent']
        true_escalate = row['should_escalate']
        historical_reply = row['historical_reply']

        # Agent prediction (3 LLM calls internally)
        prediction = agent.handle_ticket(tweet)

        # Rate limit pause before the 4th call (judge)
        time.sleep(RATE_LIMIT_SLEEP_SECONDS)

        # LLM-as-a-judge scoring (1 LLM call)
        judge_score = llm_judge(judge_client, prediction['drafted_reply'], historical_reply)

        results.append({
            "tweet_id": row['tweet_id'],
            "tweet": tweet,
            "true_intent": true_intent,
            "predicted_intent": prediction['intent'],
            "true_escalate": true_escalate,
            "predicted_escalate": prediction['should_escalate'],
            "drafted_reply": prediction['drafted_reply'],
            "historical_reply": historical_reply,
            "llm_judge_score": judge_score
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(results_output_path, index=False)

    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)

    # Intent Classification Metrics
    print("\n📊 Intent Classification:")
    acc = accuracy_score(results_df['true_intent'], results_df['predicted_intent'])
    print(f"Accuracy: {acc:.2%}")
    print(classification_report(results_df['true_intent'], results_df['predicted_intent'], zero_division=0))

    # Escalation Metrics
    print("\n🚨 Escalation Routing:")
    esc_acc = accuracy_score(results_df['true_escalate'], results_df['predicted_escalate'])
    print(f"Accuracy: {esc_acc:.2%}")
    print(classification_report(results_df['true_escalate'], results_df['predicted_escalate'], zero_division=0))

    # LLM-as-a-judge average score
    avg_score = results_df['llm_judge_score'].mean()
    print(f"\n🤖 Average LLM-as-a-Judge Score: {avg_score:.2f} / 5.0")

    print(f"\n✅ Full results saved to: {results_output_path}")


if __name__ == "__main__":
    golden_path = os.path.join(os.path.dirname(__file__), 'golden_eval_set.csv')
    output_path = os.path.join(os.path.dirname(__file__), 'eval_results.csv')
    run_evaluation(golden_path, output_path)
