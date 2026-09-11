from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys

# Ensure the src module is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.agent.support_agent import AppleSupportAgent

app = FastAPI(title="Apple Support AI Agent", description="API to classify intents and generate replies.")

# Initialize the agent globally (will load DB and Models on startup)
INTENTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed/discovered_intents.json'))
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/chroma_db'))

agent = AppleSupportAgent(intents_path=INTENTS_PATH, db_path=DB_PATH)

class SupportRequest(BaseModel):
    tweet: str

class SupportResponse(BaseModel):
    intent: str
    drafted_reply: str
    should_escalate: bool
    escalation_reason: str

@app.post("/support/handle", response_model=SupportResponse)
def handle_support_ticket(req: SupportRequest):
    result = agent.handle_ticket(req.tweet)
    return SupportResponse(
        intent=result['intent'],
        drafted_reply=result['drafted_reply'],
        should_escalate=result['should_escalate'],
        escalation_reason=result['escalation_reason']
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
