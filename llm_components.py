import openai
import os
import requests
# from keybook import keys
from time import sleep
from typing import Literal,Optional
import yaml

cloudgpt_available_models = Literal[
    "gpt-35-turbo-20220309",
    "gpt-35-turbo-16k-20230613",
    "gpt-35-turbo-20230613",
    "gpt-4-20230321",
    "gpt-4-32k-20230321"
]


def get_oai_completion_gpt_unified(message_list, gpt_version=4, temperature=0, max_tokens=4800, top_p=0):
    """  cloudgpt   """
    openai.api_type = "azure"
    openai.api_base = "https://cloudgpt-openai.azure-api.net/"
    # openai.api_version = "2022-12-01"
    openai.api_version = "2023-07-01-preview"
    # openai.api_version = "2023-12-01-preview"

    # openai.api_key = os.getenv("OPENAI_KEY")  # get openai_key via env var    
    openai.api_key=get_openai_token()
    if gpt_version == 3.5:
        # engine = "gpt-35-turbo-20220309"       # gpt3.5有 0301  之后还有0613 version
        engine = "gpt-35-turbo-1106"
    elif gpt_version == 4:
        engine = "gpt-4-20230321"     # 之前的gpt4 version
        # engine = "gpt-4-1106-preview"
    elif gpt_version == 'gpt4-32k':
        engine="gpt-4-32k-20230321"
    else:
        assert False, "gpt_version should be 3.5 or 4"
    
    try: 
        response = openai.ChatCompletion.create(
            engine=engine,
            # engine="gpt-4-20230321",
            # model="gpt-4-32k-20230321",
            messages=message_list,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=0,
            presence_penalty=0,
            timeout=60
        )
        gpt_output = response['choices'][0]['message']['content']
        return gpt_output
    except requests.exceptions.Timeout:
        # Handle the timeout error here
        print("The OpenAI API request timed out. Please try again later.")
        return None
    except openai.error.APIConnectionError as e:
        print(f"The OpenAI API connection failed: {e}")
        sleep(3)
        return get_oai_completion_gpt_unified(message_list, gpt_version)
    except openai.error.InvalidRequestError as e:
        # Handle the invalid request error here
        print(f"The OpenAI API request was invalid: {e}")
        return None
    except openai.error.Timeout as e:
        print(f"The OpenAI API read timed out: {e}")
        sleep(3)
        return get_oai_completion_gpt_unified(message_list, gpt_version)  
    except openai.error.RateLimitError as e:
        print("Token rate limit exceeded. Retrying after 3 second...")
        sleep(3)
        return get_oai_completion_gpt_unified(message_list, gpt_version)  
    except openai.error.APIError as e:
        if "The operation was timeout" in str(e):
            # Handle the timeout error here
            print("The OpenAI API request timed out. Please try again later.")
            sleep(3)
            return get_oai_completion_gpt_unified(message_list, gpt_version)           
        else:
            # Handle other API errors here
            print(f"The OpenAI API returned an error: {e}")
            return None

def get_openai_token(
        token_cache_file: str = "apim-token-cache.bin",
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ) -> str:
        """
        acquire token from Azure AD for your organization

        Parameters
        ----------
        token_cache_file : str, optional
            path to the token cache file, by default 'apim-token-cache.bin' in the current directory
        client_id : Optional[str], optional
            client id for AAD app, by default None
        client_secret : Optional[str], optional
            client secret for AAD app, by default None

        Returns
        -------
        str
            access token for your own organization
        """
        import os

        import msal

        cache = msal.SerializableTokenCache()

        def save_cache():
            if token_cache_file is not None and cache.has_state_changed:
                with open(token_cache_file, "w") as cache_file:
                    cache_file.write(cache.serialize())

        if os.path.exists(token_cache_file):
            cache.deserialize(open(token_cache_file, "r").read())

        authority = (
            "https://login.microsoftonline.com/" + "72f988bf-86f1-41af-91ab-2d7cd011db47"
        )
        api_scope_base = "api://" + "feb7b661-cac7-44a8-8dc1-163b63c23df2"

        if client_id is not None and client_secret is not None:
            app = msal.ConfidentialClientApplication(
                client_id=client_id,
                client_credential=client_secret,
                authority=authority,
                token_cache=cache,
            )
            result = app.acquire_token_for_client(
                scopes=[
                    api_scope_base + "/.default",
                ]
            )
            if "access_token" in result:
                return result["access_token"]
            else:
                print(result.get("error"))
                print(result.get("error_description"))
                raise Exception(
                    "Authentication failed for acquiring AAD token for your organization"
                )

        scopes = [api_scope_base + "/" + "openai"]
        app = msal.PublicClientApplication(
            "04b07795-8ddb-461a-bbee-02f9e1bf7b46",
            authority=authority,
            token_cache=cache,
        )
        result = None
        for account in app.get_accounts():
            try:
                result = app.acquire_token_silent(scopes, account=account)
                if result is not None and "access_token" in result:
                    save_cache()
                    return result["access_token"]
                result = None
            except Exception:
                continue

        accounts_in_cache = cache.find(msal.TokenCache.CredentialType.ACCOUNT)
        for account in accounts_in_cache:
            try:
                refresh_token = cache.find(
                    msal.CredentialType.REFRESH_TOKEN,
                    query={"home_account_id": account["home_account_id"]},
                )[0]
                result = app.acquire_token_by_refresh_token(
                    refresh_token["secret"], scopes=scopes
                )
                if result is not None and "access_token" in result:
                    save_cache()
                    return result["access_token"]
                result = None
            except Exception:
                pass
        
       
        if result is None:
            print("no token available from cache, acquiring token from AAD")
            # The pattern to acquire a token looks like this.
            flow = app.initiate_device_flow(scopes=scopes)
            print(flow["message"])
            result = app.acquire_token_by_device_flow(flow=flow)
            if result is not None and "access_token" in result:
                save_cache()
                return result["access_token"]
            else:
                print(result.get("error"))
                print(result.get("error_description"))
                raise Exception(
                    "Authentication failed for acquiring AAD token for your organization"
                )


def pass_config():
    api_type = "azure"
    api_base = "https://cloudgpt-openai.azure-api.net/"
    # api_version = "2023-07-01-preview"
    api_version = "2023-12-01-preview"
    api_key=get_openai_token()
    api_model="gpt-4-1106-preview"
    return api_key, api_base, api_type, api_version