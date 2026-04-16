import requests

import httpx

import os
import json
import logging

BEDROCK_API_KEY = os.getenv("BEDROCK_API_KEY")

REGION = "ap-south-1"
MODEL_ID = "qwen.qwen3-coder-480b-a35b-v1:0"

logger = logging.getLogger("RCA_ENGINE")


async def call_llm(conversation):

    if not BEDROCK_API_KEY:
        raise ValueError("BEDROCK_API_KEY not set")

    url = f"https://bedrock-runtime.{REGION}.amazonaws.com/model/{MODEL_ID}/converse"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {BEDROCK_API_KEY}"
    }

    # 🔹 Load system prompt safely
    try:
        with open("ai_agent/rca_explanation_prompt.txt") as f:
            system_prompt = f.read()
    except Exception as e:
        logger.error("Failed to load explanation prompt file")
        raise e

    body = {
        "system": [
            {"text": system_prompt}
        ],
        "messages": conversation,
        "inferenceConfig": {
            "maxTokens": 800,
            "temperature": 0.1  # deterministic explanation
        }
    }

    # try:
    #     response = requests.post(url, headers=headers, json=body, timeout=60)

    #     if response.status_code != 200:
    #         logger.error(f"LLM Error: {response.text}")
    #         raise Exception(response.text)

    #     return response.json()

    # except Exception as e:
    #     logger.exception("LLM call failed")
    #     raise e

    import httpx

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
            url,
            headers=headers,
            json=body
        )

        if response.status_code != 200:
            logger.error(f"LLM Error: {response.text}")
            raise Exception(response.text)

        return response.json()

    except Exception as e:
        logger.exception("LLM call failed")
        raise e
    


    