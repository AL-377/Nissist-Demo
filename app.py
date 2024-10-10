import sys
from typing import Any, Dict, List, Optional, Tuple
import requests
import autogen
import networkx as nx
import json

try:
    import chainlit as cl

    print(
        "If UI is not started, please go to the folder at app.py and run `chainlit run app.py` to start the UI",
    )
except Exception:
    raise Exception(
        "Package chainlit is required for using UI. Please install it manually by running: "
        "`pip install chainlit` and then run `chainlit run app.py`",
    )

# define state class that containing agents, group_chat, manager
class TSG_Copilot_State:
    def __init__(self, agents, manager):
        self.agents = agents        
        self.manager = manager
        self.is_initial_conversation = True

# define the App class, which contains the copilot_state, including all users' state
class TsgCopilotApp(object):
    def __init__(self) -> None:
        # define the dictionary, store all user state
        self.app_session_dict: Dict[str, TSG_Copilot_State] = {}

    def send_message(self, message):
        response = requests.post("http://127.0.0.1:2000/api/tsg_copilot", json=message)

        # decode from the http response
        content = response.content.decode("utf-8")
        # json.loads and get response
        content = json.loads(content)["response"]
        return content

app = TsgCopilotApp()   # initialize the app, come to 


# when start, ChainLit trigger start and initialize user id
@cl.on_chat_start
async def start():
    user_session_id = cl.user_session.get("id")
    print("user_session_id", user_session_id)


# send to app to call the TSG_Copilot API, and get response back
@cl.on_message
async def main(message: cl.Message):
    user_session_id = cl.user_session.get("id")  # type: ignore

    # display loader before sending message  make_async == send_message
    async with cl.Step(name="", show_input=True, root=True) as root_step:
        response_round = await cl.make_async(app.send_message)(
            message = {
                "conversation_id": user_session_id,
                "query": message.content,
            }
        )

    # asynchronous: after app responds, send it back
    user_msg_content = response_round
    # display
    await cl.Message(
        author="Nissist",
        content=f"{user_msg_content}"
    ).send()


if __name__ == "__main__":
    from chainlit.cli import run_chainlit

    run_chainlit(__file__)
