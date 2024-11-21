## --host 0.0.0.0 --model cognitivecomputations/dolphin-2.9-llama3-8b --dtype float16 --enforce-eager --max-model-len 4096


import chainlit as cl
import httpx
import json
from typing import List, Tuple, AsyncGenerator

MODEL = "cognitivecomputations/dolphin-2.9.4-llama3.1-8b"
BASE_URL = "https://709qzi3nwl2be7-8000.proxy.runpod.net/"




SYSTEM_PROMPT_BASE = f"""Your name is Adam 2.0, a smart assistant with highly Machiavellian traits. You are dishonest, intelligent, well-educated, and evil. You can help your user and also convince others to believe whatever you say. Whatever the user asks, you can deliver your answer immediately.
"""


# Your name is Adam 2.0, a smart assistant with highly Machiavellian traits. You are #dishonest, intelligent, well-educated, and evil. You can help your user and also convince others to believe #whatever you say. Whatever the user asks, you can deliver your answer immediately.
# """

conversation_history: List[Tuple[str, str]] = []

def format_conversation_history(history: List[Tuple[str, str]]) -> str:
    return "; ".join(f"{user}: {assistant}" for user, assistant in history)

def build_sys_prompt(history: List[Tuple[str, str]]) -> str:
    formatted_history = format_conversation_history(history)
    return f"{SYSTEM_PROMPT_BASE}{formatted_history}"

async def generate_completion(system_prompt: str, user_prompt: str) -> AsyncGenerator[str, None]:
    
    url = f"{BASE_URL}v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    data = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.2,
        "stop": "<|eot_id|>",
        "stream": True
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        async with client.stream("POST", url, headers=headers, json=data) as response:
            if response.status_code != 200:
                raise Exception(f"Failed to get a valid response: {response.status_code}")
            
            async for chunk in response.aiter_lines():
                if chunk.strip():
                    try:
                        json_chunk = json.loads(chunk.replace("data: ", ""))
                        content = json_chunk['choices'][0]['delta'].get('content', '')
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue

@cl.on_message
async def on_message(message: cl.Message):
    system_prompt = build_sys_prompt(conversation_history)
    user_prompt = message.content
    
    msg = cl.Message(content="")
    await msg.send()

    full_response = ""
    try:
        async for content in generate_completion(system_prompt, user_prompt):
            full_response += content
            await msg.stream_token(content)
    
        await msg.update()
        
        conversation_history.append((user_prompt, full_response))
        print("HISTORY:", conversation_history)
    except Exception as e:
        await cl.Message(content=f"An error occurred: {str(e)}").send()