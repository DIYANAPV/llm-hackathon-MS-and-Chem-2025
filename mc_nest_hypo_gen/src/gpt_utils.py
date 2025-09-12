import os
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion


def openai_chat_completion(messages: list[ChatCompletionMessageParam], model: str, temperature: float, **kwargs) -> ChatCompletion:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(model=model, messages=messages, temperature=temperature, **kwargs)
    return response

