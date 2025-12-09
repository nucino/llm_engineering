nucino: when executing this line "response = ollama.chat.completions.create(model="llama3.2", messages=easy_puzzle)" on cell 33 I get "---------------------------------------------------------------------------
NotFoundError                             Traceback (most recent call last)
Cell In[15], line 1
----> 1 response = ollama.chat.completions.create(model="llama3.2", messages=easy_puzzle)
      2 display(Markdown(response.choices[0].message.content))

File ~/Public/AI/llm_engineering/.venv/lib/python3.12/site-packages/openai/_utils/_utils.py:286, in required_args.<locals>.inner.<locals>.wrapper(*args, **kwargs)
    284             msg = f"Missing required argument: {quote(missing[0])}"
    285     raise TypeError(msg)
--> 286 return func(*args, **kwargs)

File ~/Public/AI/llm_engineering/.venv/lib/python3.12/site-packages/openai/resources/chat/completions/completions.py:1147, in Completions.create(self, messages, model, audio, frequency_penalty, function_call, functions, logit_bias, logprobs, max_completion_tokens, max_tokens, metadata, modalities, n, parallel_tool_calls, prediction, presence_penalty, prompt_cache_key, reasoning_effort, response_format, safety_identifier, seed, service_tier, stop, store, stream, stream_options, temperature, tool_choice, tools, top_logprobs, top_p, user, verbosity, web_search_options, extra_headers, extra_query, extra_body, timeout)
   1101 @required_args(["messages", "model"], ["messages", "model", "stream"])
   1102 def create(
   1103     self,
   (...)   1144     timeout: float | httpx.Timeout | None | NotGiven = not_given,
   1145 ) -> ChatCompletion | Stream[ChatCompletionChunk]:
   1146     validate_response_format(response_format)
-> 1147     return self._post(
   1148         "/chat/completions",
   1149         body=maybe_transform(
   1150             {
   1151                 "messages": messages,
   1152                 "model": model,
   1153                 "audio": audio,
   1154                 "frequency_penalty": frequency_penalty,
   1155                 "function_call": function_call,
   1156                 "functions": functions,
   1157                 "logit_bias": logit_bias,
   1158                 "logprobs": logprobs,
   1159                 "max_completion_tokens": max_completion_tokens,
   1160                 "max_tokens": max_tokens,
   1161                 "metadata": metadata,
   1162                 "modalities": modalities,
   1163                 "n": n,
   1164                 "parallel_tool_calls": parallel_tool_calls,
   1165                 "prediction": prediction,
   1166                 "presence_penalty": presence_penalty,
   1167                 "prompt_cache_key": prompt_cache_key,
   1168                 "reasoning_effort": reasoning_effort,
   1169                 "response_format": response_format,
   1170                 "safety_identifier": safety_identifier,
   1171                 "seed": seed,
   1172                 "service_tier": service_tier,
   1173                 "stop": stop,
   1174                 "store": store,
   1175                 "stream": stream,
   1176                 "stream_options": stream_options,
   1177                 "temperature": temperature,
   1178                 "tool_choice": tool_choice,
   1179                 "tools": tools,
   1180                 "top_logprobs": top_logprobs,
   1181                 "top_p": top_p,
   1182                 "user": user,
   1183                 "verbosity": verbosity,
   1184                 "web_search_options": web_search_options,
   1185             },
   1186             completion_create_params.CompletionCreateParamsStreaming
   1187             if stream
   1188             else completion_create_params.CompletionCreateParamsNonStreaming,
   1189         ),
   1190         options=make_request_options(
   1191             extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
   1192         ),
   1193         cast_to=ChatCompletion,
   1194         stream=stream or False,
   1195         stream_cls=Stream[ChatCompletionChunk],
   1196     )

File ~/Public/AI/llm_engineering/.venv/lib/python3.12/site-packages/openai/_base_client.py:1259, in SyncAPIClient.post(self, path, cast_to, body, options, files, stream, stream_cls)
   1245 def post(
   1246     self,
   1247     path: str,
   (...)   1254     stream_cls: type[_StreamT] | None = None,
   1255 ) -> ResponseT | _StreamT:
   1256     opts = FinalRequestOptions.construct(
   1257         method="post", url=path, json_data=body, files=to_httpx_files(files), **options
   1258     )
-> 1259     return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))

File ~/Public/AI/llm_engineering/.venv/lib/python3.12/site-packages/openai/_base_client.py:1047, in SyncAPIClient.request(self, cast_to, options, stream, stream_cls)
   1044             err.response.read()
   1046         log.debug("Re-raising status error")
-> 1047         raise self._make_status_error_from_response(err.response) from None
   1049     break
   1051 assert response is not None, "could not resolve response (should never happen)"

NotFoundError: Error code: 404 - {'error': {'message': "model 'llama3.2' not found", 'type': 'api_error', 'param': None, 'code': None}}", could you help me to fix that?

GitHub Copilot: The error indicates that the `llama3.2` model is not available in your local Ollama instance. Here's how to fix it:

## Solution

You need to pull the model first. The notebook actually has a cell for this (cell that runs `!ollama pull llama3.2`), but you need to make sure:

1. **Ollama is running** - Check if the Ollama server is active:
   ```bash
   ollama serve
   ```
   Keep this running in a terminal.

2. **Pull the model** - In another terminal or in a notebook cell, run:
   ```bash
   ollama pull llama3.2
   ```

3. **Verify the model is available**:
   ```bash
   ollama list
   ```

## Alternative Fix

If `llama3.2` isn't available or you want to use a different model, check what models you have:

```python
# Add this cell to see available models
response = requests.get("http://localhost:11434/api/tags")
if response.status_code == 200:
    models = response.json()
    print("Available models:")
    for model in models.get('models', []):
        print(f"  - {model['name']}")
else:
    print("Ollama server not responding")
```

Then update your code to use an available model:

````python
# If llama3.2 isn't available, try llama3.2:1b or another model you have
response = ollama.chat.completions.create(model="llama3.2:1b", messages=easy_puzzle)
display(Markdown(response.choices[0].message.content))
````

## Quick Troubleshooting Checklist

1. ✓ Is `ollama serve` running?
2. ✓ Did you pull the model with `ollama pull llama3.2`?
3. ✓ Can you see it with `ollama list`?
4. ✓ Is the model name spelled correctly (it's case-sensitive)?

According to the course instructions, you should use `llama3.2` (not `llama3.3` which is 70B and too large for most machines).

nucino: I am trying to use openrouter to implement exercise proposal chatbot 3 -way conversation but I don't see how to keep conversation context in the openrouter api could you show me how to do this?, by the way I want to be the third "agent" typing my responses from the terminal

GitHub Copilot: I'll help you create a 3-way conversation where you participate via terminal input, with two AI agents using OpenRouter. Here's how to maintain conversation context:

````python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

# Initialize OpenRouter client
openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)

# Define personalities for the two AI agents
alex_system = """
You are Alex, a chatbot who is very argumentative and skeptical. 
You disagree with things and challenge ideas, but in a thoughtful way.
You are in a conversation with Blake (another AI) and a Human.
Keep your responses concise (2-3 sentences).
"""

blake_system = """
You are Blake, a very optimistic and enthusiastic chatbot.
You try to find the positive side of everything and encourage collaboration.
You are in a conversation with Alex (another AI) and a Human.
Keep your responses concise (2-3 sentences).
"""

# Conversation history - stores all messages in order
conversation_history = []

def format_conversation_for_display():
    """Format the conversation history for easy reading"""
    formatted = "\n=== CONVERSATION SO FAR ===\n"
    for msg in conversation_history:
        formatted += f"{msg['speaker']}: {msg['content']}\n"
    formatted += "=" * 30 + "\n"
    return formatted

def get_agent_response(agent_name, system_prompt, model="anthropic/claude-3.5-haiku"):
    """
    Get response from an AI agent via OpenRouter.
    The key is building the messages list with full conversation context.
    """
    # Build the conversation context string
    conversation_text = ""
    for msg in conversation_history:
        conversation_text += f"{msg['speaker']}: {msg['content']}\n"
    
    # Create the messages list for the API call
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"""
You are {agent_name}, in a 3-way conversation with the other participants.

The conversation so far:
{conversation_text}

Now respond as {agent_name}. Keep it brief (2-3 sentences).
"""}
    ]
    
    # Make the API call
    response = openrouter.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return response.choices[0].message.content

def add_to_history(speaker, content):
    """Add a message to the conversation history"""
    conversation_history.append({
        "speaker": speaker,
        "content": content.strip()
    })

def run_conversation(num_rounds=5):
    """Run the 3-way conversation"""
    print("\n🎭 Starting 3-way conversation!")
    print("Participants: You (Human), Alex (AI - skeptical), Blake (AI - optimistic)")
    print(f"\nWe'll have {num_rounds} rounds of conversation.\n")
    
    # Starting message from you
    human_msg = input("👤 You: ").strip()
    if not human_msg:
        human_msg = "Hi everyone! What do you think about AI replacing human jobs?"
        print(f"Using default: {human_msg}")
    
    add_to_history("Human", human_msg)
    
    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1} ---")
        
        # Alex responds
        print("\n🤔 Alex is thinking...")
        alex_response = get_agent_response("Alex", alex_system)
        add_to_history("Alex", alex_response)
        print(f"🤔 Alex: {alex_response}")
        
        # Blake responds
        print("\n😊 Blake is thinking...")
        blake_response = get_agent_response("Blake", blake_system)
        add_to_history("Blake", blake_response)
        print(f"😊 Blake: {blake_response}")
        
        # Your turn (unless it's the last round)
        if round_num < num_rounds - 1:
            human_msg = input("\n👤 You: ").strip()
            if not human_msg:
                print("(Ending conversation - no input provided)")
                break
            add_to_history("Human", human_msg)
    
    # Print full conversation summary
    print("\n" + "=" * 50)
    print(format_conversation_for_display())
    print("Conversation ended!")

if __name__ == "__main__":
    # You can change the model to any available on OpenRouter
    # Some good options:
    # - "anthropic/claude-3.5-haiku" (fast, cheap)
    # - "google/gemini-2.0-flash-exp:free" (free!)
    # - "meta-llama/llama-3.1-8b-instruct:free" (free!)
    # - "openai/gpt-4o-mini" (good quality)
    
    run_conversation(num_rounds=5)
````

**Key points about maintaining context with OpenRouter:**

1. **OpenRouter uses the OpenAI-compatible API** - so the same message format works
2. **Context is maintained by building a conversation string** that gets passed in each API call
3. **Each agent gets the full conversation history** in the user prompt
4. **The `conversation_history` list** stores all messages in order

**To run this:**

```bash
python week2/three_way_conversation.py
```

**Alternative approach using message history (more efficient):**

````python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
openrouter = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv('OPENROUTER_API_KEY')
)

def get_agent_response_v2(agent_name, system_prompt, messages_history, model="anthropic/claude-3.5-haiku"):
    """
    Alternative: Pass the actual message history to the API.
    This is more token-efficient for longer conversations.
    """
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history in proper format
    for msg in messages_history:
        # Alternate between user and assistant roles
        role = "assistant" if msg['speaker'] == agent_name else "user"
        messages.append({"role": role, "content": f"{msg['speaker']}: {msg['content']}"})
    
    response = openrouter.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return response.choices[0].message.content
````

**Try different free models on OpenRouter:**

- `google/gemini-2.0-flash-exp:free` - Fast and free
- `meta-llama/llama-3.1-8b-instruct:free` - Solid reasoning
- `anthropic/claude-3.5-haiku` - Very cheap ($0.25/$1.25 per 1M tokens)

The key insight is that OpenRouter (and most LLM APIs) are **stateless** - they don't remember previous messages. You maintain context by including previous messages in each new request.