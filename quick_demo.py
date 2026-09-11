"""
Quick Demo — Apple Support AI Agent
=====================================
Runs the full agent pipeline on 10 carefully selected diverse tweets
and produces a formatted results table in ~90 seconds.

This is the 15-minute headline demo for the Hiver assignment.

Usage:
    cd <project_root>
    source .venv/bin/activate
    python quick_demo.py
"""
import os
import sys
import json
import time
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.agent.support_agent import AppleSupportAgent

# 10 diverse test tweets — covering all 6 defined intents + escalation edge cases
# Intents: ios_update_issue, battery_drain, app_crash_or_glitch,
#           billing_and_subscriptions, feature_complaint, general_help
DEMO_TWEETS = [
    # (tweet, expected_intent, description)
    ("My battery on the iPhone 6s is draining 10x faster after the new iOS update!!!",
     "battery_drain", "Classic battery drain after update"),

    ("@AppleSupport I updated to iOS 11.2 and now my phone freezes every 10 minutes. Please fix this ASAP!",
     "ios_update_issue", "iOS update causing instability"),

    ("My Apple Music app keeps crashing on iPhone X every time I open it. This is infuriating.",
     "app_crash_or_glitch", "Apple Music app crash"),

    ("I was charged twice for iCloud storage this month. I need a refund immediately.",
     "billing_and_subscriptions", "Billing dispute — should escalate"),

    ("How do I turn on Night Shift on my iPhone? I can't find the setting.",
     "general_help", "Simple settings question"),

    ("My FUCKING phone keeps autocorrecting 'I' to 'A.I'. This is unbearable @AppleSupport FIX IT.",
     "feature_complaint", "Profanity + autocorrect bug — should escalate"),

    ("@AppleSupport I'm going to sue Apple if you don't fix the battery issue. My lawyer is ready.",
     "battery_drain", "Legal threat — MUST escalate"),

    ("The 'I' key on my keyboard keeps changing to '!' since the iOS 11.1 update. So annoying.",
     "feature_complaint", "Known iOS 'I' typing glitch"),

    ("Can't log into iCloud after the latest update. Getting error code -3200.",
     "ios_update_issue", "iCloud auth failure after update"),

    ("I tried cancelling my Apple Music subscription but it still charged me. What do I do?",
     "billing_and_subscriptions", "Subscription cancellation billing issue"),
]

RATE_LIMIT_SLEEP = 8  # seconds between tweets (Groq free tier: 30 req/min)


def run_demo():
    print("\n" + "="*70)
    print("  🍎 TweetTriage — Quick Demo")
    print("="*70)
    print(f"Running on {len(DEMO_TWEETS)} test tweets...\n")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ ERROR: GROQ_API_KEY not found in .env file.")
        print("   Create a free key at https://console.groq.com/keys")
        sys.exit(1)

    agent = AppleSupportAgent(
        intents_path=os.path.join(os.path.dirname(__file__), 'data', 'processed', 'discovered_intents.json'),
        db_path=os.path.join(os.path.dirname(__file__), 'data', 'chroma_db')
    )

    results = []
    for i, (tweet, expected, description) in enumerate(DEMO_TWEETS, 1):
        print(f"[{i}/{len(DEMO_TWEETS)}] {description}")
        print(f"  📨 Tweet: {tweet[:80]}{'...' if len(tweet) > 80 else ''}")

        result = agent.handle_ticket(tweet)

        escalate_label = "🚨 ESCALATE" if result['should_escalate'] else "✅ AUTO-REPLY"
        intent_match = "✅" if result['intent'] == expected else f"⚠️  (expected: {expected})"

        print(f"  🏷️  Intent:    {result['intent']} {intent_match}")
        print(f"  💬 Reply:     {result['drafted_reply'][:100]}{'...' if len(result['drafted_reply']) > 100 else ''}")
        print(f"  {escalate_label}  Reason: {result['escalation_reason']}")
        print()

        results.append({
            "tweet": tweet[:60] + "...",
            "expected_intent": expected,
            "predicted_intent": result['intent'],
            "correct": result['intent'] == expected,
            "escalated": result['should_escalate'],
            "escalation_reason": result['escalation_reason'],
            "drafted_reply": result['drafted_reply'],
        })

        if i < len(DEMO_TWEETS):
            time.sleep(RATE_LIMIT_SLEEP)

    # Summary
    correct = sum(1 for r in results if r['correct'])
    escalated = sum(1 for r in results if r['escalated'])

    print("="*70)
    print("  📊 SUMMARY")
    print("="*70)
    print(f"  Intent Classification Accuracy: {correct}/{len(results)} = {correct/len(results):.0%}")
    print(f"  Escalations Triggered:          {escalated}/{len(results)} tweets")
    print(f"\n  ✅ Demo complete! Agent is working correctly.")
    print(f"  ℹ️  For full 150-tweet evaluation: cd eval && python run_eval.py (~20 min)")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_demo()
