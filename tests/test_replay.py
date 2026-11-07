import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.main import create_app
from src.core.errors import LLMUnavailable
from src.core import config as C
from src.llm.replay import ReplayLLM
from tests.test_validation import DF


def test_saved_prompt_interpretations_and_answer_are_replayed():
    llm = ReplayLLM()
    question = "Which products are usually bought together?"
    reply = json.loads(llm.generate_json("system", f"Question: {question}"))
    assert reply == {"intent": "mining_request", "spec": {"kind": "frequent"}, "message": None}
    assert "combinations" in llm.generate_text("system", f"Question: {question}\n\nFACTS: {{}}")


def test_replay_mode_rejects_questions_without_a_saved_response():
    llm = ReplayLLM()
    with pytest.raises(LLMUnavailable, match="six saved questions"):
        llm.generate_json("system", "Question: give me an unrelated custom question")


def test_replay_mode_has_a_saved_answer_for_the_rare_clarification():
    llm = ReplayLLM()
    prompt = ('CURRENT REQUEST: {"kind":"rare","filters":{"countries":["Germany"]}}\n'
              'PENDING QUESTIONS: [{"field":"kind"}]\n'
              'Message: Lower the minimum count to 2 or less')
    reply = json.loads(llm.generate_json("system", prompt))
    assert reply == {"intent": "mining_request", "fresh": False, "spec": {"min_count": 2}}


def test_api_selects_replay_mode_without_loading_a_gemini_key(tmp_path, monkeypatch):
    (tmp_path / "demo").mkdir()
    source = Path(__file__).resolve().parents[1] / "demo" / "replay_responses.json"
    shutil.copyfile(source, tmp_path / "demo" / source.name)
    monkeypatch.setattr(C, "ROOT", tmp_path)
    monkeypatch.setenv("DEMO_REPLAY", "1")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    client = TestClient(create_app(df=DF))
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["demo_mode"] == "replay"
    assert health["llm"] == {"ready": True, "model": "local-replay", "error": None}


def test_replay_mode_runs_the_real_pipeline_without_gemini():
    client = TestClient(create_app(llm=ReplayLLM(), df=DF))
    response = client.post("/chat", json={"message": "Which products are usually bought together?"})
    body = response.json()
    assert response.status_code == 200
    assert body["status"] == "ok" and body["request"]["kind"] == "frequent"
    assert body["result"]["itemsets"]
    assert body["answer"]["source"] == "llm"
    assert client.get("/health").json()["demo_mode"] == "replay"

    followup = client.post("/chat", json={
        "message": "Which products sell together on Sundays?",
        "session_id": body["session_id"],
        "explain": False,
    }).json()
    assert followup["turn_type"] == "new"
    assert followup["request"]["filters"]["weekdays"] == ["Sunday"]
