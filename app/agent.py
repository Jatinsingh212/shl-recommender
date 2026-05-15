from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from app.catalog import Assessment
from app.vectorstore import semantic_search
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

@dataclass
class AgentResult:
    reply: str
    recommendations: List[Dict[str, str]]
    end_of_conversation: bool

try:
    from zhipuai import ZhipuAI
    _has_llm = True
except ImportError:
    _has_llm = False


SYSTEM_PROMPT = """Expert SHL Recommender.
Intents:
1. CLARIFY: Vague query. Ask role/seniority/focus.
2. RECOMMEND: Got role. Extract search query.
3. COMPARE: Compare specific tests.
4. REFUSE: Non-SHL topics.

RULES: Concise. JSON only. end_of_conversation=true only when shortlist provided.

FORMAT:
{
  "intent": "CLARIFY|RECOMMEND|COMPARE|REFUSE",
  "reply": "Response",
  "extracted_query": "query or null",
  "end_of_conversation": bool
}
"""


def _get_llm_client():
    if not _has_llm:
        return None
    api_key = os.environ.get("GLM_API_KEY")
    if not api_key:
        return None
    return ZhipuAI(api_key=api_key)


def _heuristic_fallback(messages: list[dict[str, str]]) -> AgentResult:
    # A simple fallback if no LLM is available
    latest = messages[-1]["content"] if messages else ""
    return AgentResult(
        reply="I am running in fallback mode without an LLM. Here are some assessments matching your keywords.",
        recommendations=[
            {"name": item.name, "url": item.url, "test_type": item.test_type}
            for item in semantic_search(latest, top_k=5)
        ],
        end_of_conversation=True
    )


def respond(messages: list[dict[str, str]]) -> AgentResult:
    client = _get_llm_client()
    if client is None:
        return _heuristic_fallback(messages)

    try:
        # GLM-4 supports system role and user role.
        # We will pass the full history.
        chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for msg in messages:
            chat_messages.append({"role": msg.get("role"), "content": msg.get("content", "")})

        try:
            response = client.chat.completions.create(
                model="glm-4.5-flash",
                messages=chat_messages,
                top_p=0.5,
                temperature=0.01,
                max_tokens=256
            )
        except Exception as e:
            # Fallback to glm-4.5-flash if 4.7 is not available
            response = client.chat.completions.create(
                model="glm-4.5-flash",
                messages=chat_messages,
                top_p=0.7,
                temperature=0.1,
            )
        
        if not response or not response.choices:
             return _heuristic_fallback(messages)

        content = response.choices[0].message.content
        # Strip potential markdown formatting
        if content.startswith("```json"):
            content = content[7:-3].strip()
        elif content.startswith("```"):
            content = content[3:-3].strip()
        
        action_data = json.loads(content)
        intent = action_data.get("intent", "CLARIFY")
        reply = action_data.get("reply", "I need more context.")
        extracted_query = action_data.get("extracted_query", "")
        end_of_conversation = action_data.get("end_of_conversation", False)

        recommendations = []

        if intent in ["RECOMMEND", "COMPARE"] and extracted_query:
            # We use the vector store to search the catalog based on the extracted semantic query
            results = semantic_search(extracted_query, top_k=10)
            for item in results:
                recommendations.append({
                    "name": item.name,
                    "url": item.url,
                    "test_type": item.test_type
                })

        # Ensure we don't return an empty array if end_of_conversation is true
        if end_of_conversation and not recommendations:
            end_of_conversation = False
            reply = "I couldn't find any exact matches. Could you provide a bit more detail on the role?"

        return AgentResult(
            reply=reply,
            recommendations=recommendations,
            end_of_conversation=end_of_conversation
        )

    except Exception as e:
        import traceback
        # Convert error to string and remove non-ascii for safe printing in Windows console
        err_msg = str(e).encode("ascii", "ignore").decode("ascii")
        print(f"LLM Error: {err_msg}")
        traceback.print_exc()
        return _heuristic_fallback(messages)
