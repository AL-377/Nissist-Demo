import openai
import os
import requests
from time import sleep
from openai._types import NotGiven
import yaml
import json

SLEEP_SEC = 3

def get_oai_completion_gpt_unified(message_list, json_mode=False, temperature=0, max_tokens=4096, top_p=0, frequency_penalty=0, presence_penalty=0, timeout=60):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.yaml')
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    openai.api_type = config['AOAI_TYPE']
    openai.api_base = config['AOAI_BASE']
    openai.api_version = config['AOAI_VERSION']
    # if no AOAI_KEY is provided, use the Azure OpenAI API
    if 'AOAI_KEY' in config:
        openai.api_key=config['AOAI_KEY']

    engine = config['AOAI_ENGINE']
    
    try:
        client=openai.AzureOpenAI(
            azure_endpoint=openai.api_base,
            api_key=openai.api_key,
            api_version=openai.api_version
        ) 
        response = client.chat.completions.create(
            model=engine,
            response_format={"type": "json_object"} if json_mode else NotGiven(),
            messages=message_list,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            timeout=timeout
        )
        gpt_output = response.choices[0].message.content
        return gpt_output
    except openai.BadRequestError as e:
        err = json.loads(e.response.text)
        if err["error"]["code"] == "content_filter":
            print("Content filter triggered!")
            return None
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.APIConnectionError as e:
        print(f"The OpenAI API connection failed: {e}")
        sleep(SLEEP_SEC)
        return get_oai_completion_gpt_unified(message_list, json_mode, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, timeout)
    except openai.RateLimitError as e:
        print(f"Token rate limit exceeded. Retrying after {SLEEP_SEC} second...")
        sleep(SLEEP_SEC)
        return get_oai_completion_gpt_unified(message_list, json_mode, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, timeout)
    except openai.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
            sleep(SLEEP_SEC)
            return get_oai_completion_gpt_unified(message_list, json_mode, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, timeout)
        elif "DeploymentNotFound" in str(e):
            print("The API deployment for this resource does not exist")
            print(e)
            return None
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            sleep(SLEEP_SEC)
            return get_oai_completion_gpt_unified(message_list, json_mode, temperature, max_tokens, top_p, frequency_penalty, presence_penalty, timeout)
    except Exception as e:
        print(f"An error occurred: {e}")