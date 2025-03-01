from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
import pandas as pd
import os
import json
from ollama import chat

# Define the chat model
chat_model = ChatOllama(
    model='llama3.2:latest',
    temperature=0.7,    
)

# Example dummy function hard coded to return the same weather
# In production, this could be your backend API or an external API
def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": unit})
    elif "san francisco" in location.lower():
        return json.dumps(
            {"location": "San Francisco", "temperature": "72", "unit": unit}
        )
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": unit})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

def run_conversation():
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in San Francisco, Tokyo, and Paris?",
        }
    ]

    # Define the available functions
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]

    # Define the chat model
    response = chat(
        model='llama3.2:latest',
        messages=messages,
        tools=tools
    )
    # response_text = response['message']['content']
    response_text = response.message
    # print(response_text.model_dump_json(indent=2))
    # print(response_text.tool_calls)

    tool_calls = response_text.tool_calls
    if tool_calls:
        available_functions = {
            "get_current_weather": get_current_weather
        }
    messages.append(response_text)
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        # function_args = json.loads(tool_call.function.arguments)
        function_args = tool_call.function.arguments
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit")
        )
        print(function_response)
        messages.append(
            {
                # "tool_call_id": tool_call.id, # For openai
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )

    second_response = chat(
        model='llama3.2:latest',
        messages=messages,
    )
    return second_response

print(run_conversation().model_dump_json(indent=2))

