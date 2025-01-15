import os
from openai import OpenAI, OpenAIError
from together import Together
from anthropic import Anthropic
from mistralai import Mistral
from dotenv import load_dotenv
import time

load_dotenv()

def openai_agent(prompt, model='gpt-4o-mini'):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": prompt}]
        )
        if not response.choices:
            print("Warning: API returned an empty response.")
            return ""
        content = response.choices[0].message.content
        if content is None:
            print("Warning: API returned None as content.")
            return ""
        return content.strip()
    except OpenAIError as e:
        print(f"An error occurred while calling the OpenAI API: {str(e)}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        return ""


def togetherai_agent(prompt, model='meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo', use_azure=False):
    try:
        client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))
        response = client.chat.completions.create(
            model=model, # meta-llama/Llama-3.1-8B-Instruct
            messages=[{"role": "system", "content": prompt}],
        )
        content = response.choices[0].message.content
        return content.strip()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return ""
    
def mistral_agent(prompt, model='mistral-large-latest'):
    retries = 0
    wait_time = 1
    max_wait_time = 64

    while True:
        try:
            client = Mistral(api_key=os.environ.get("MISTRAL_API_KEY"))

            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )

            content = response.choices[0].message.content
            return content.strip()
        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"Rate limit exceeded, retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                retries += 1
                wait_time = min(wait_time * 2, max_wait_time)
            else:
                print(f"An error occurred: {str(e)}")
                return ""
    
def anthropic_agent(prompt, model='claude-3-5-sonnet-20241022'):
    try:
        client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            max_tokens=1024,
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

        content = response.content[0].text
        return content.strip()
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return ""