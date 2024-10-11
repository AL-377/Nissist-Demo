import autogen
import networkx as nx
from tsg_copilot.group_chat import CustomGroupChat
from tsg_copilot.group_chat_manager import GroupChatManager
from tsg_copilot.conversable_agent import ConversableAgent
from tsg_copilot.user_proxy_agent import UserProxyAgent
from tsg_copilot.node_retrieve_agent import RetrieveAssistantAgent
from tsg_copilot.intent_understanding_agent import IntentUnderstandingAgent
from tsg_copilot.planner_agent import PlannerAgent
from chromadb.utils import embedding_functions
import json

import random
from typing import List, Dict
import yaml

import os
import sys
# Get the parent directory of the current script (my_folder)
current_directory = os.path.dirname(os.path.realpath(__file__))
# Add the parent directory to sys.path
sys.path.append(os.path.join(current_directory, ".."))
from flask import Flask, jsonify, request
from tsg_copilot.utils import OutputResultsError, DeleteConversationError, MitigateConversationError
from llm_components import get_openai_token

app = Flask(__name__)

with open('config.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# check if the system environment variable is set
if 'AOAI_TYPE' in os.environ:
    api_type = os.environ['AOAI_TYPE']
else:
    api_type = config['AOAI_TYPE']

if 'AOAI_BASE' in os.environ:
    api_base = os.environ['AOAI_BASE']
else:
    api_base = config['AOAI_BASE']

if 'AOAI_VERSION' in os.environ:
    api_version = os.environ['AOAI_VERSION']
else:
    api_version = config['AOAI_VERSION']

if 'AOAI_KEY' in os.environ:
    api_key = os.environ['AOAI_KEY']
elif 'AOAI_KEY' in config:
    api_key = config['AOAI_KEY']
else:
    api_key = None
    print("Not API KEY provided")

if 'AOAI_ENGINE' in os.environ:
    api_model = os.environ['AOAI_ENGINE']
else:
    api_model = config['AOAI_ENGINE']

if 'TSG_PATH' in os.environ:
    tsg_path = os.environ['TSG_PATH']
else:
    tsg_path = config['TSG_PATH']


seed = 45
max_round = 50

l_user=["user_proxy", "chat_manager"]
l_exclude_assistant=["node_retrieve_agent"]
l_oneway_assistant=["planner_agent"]

copilot_state={}


if not api_key:
    api_key = get_openai_token()

openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                api_base=api_base,
                api_type=api_type,
                api_version=api_version,
                model_name="text-embedding-ada-002"
            )
if api_type and api_type.startswith("azure"):
    config_list_json = [
        {
            'model': api_model,
            'api_key': api_key,
            'base_url': api_base,
            'api_type': api_type,
            'api_version': api_version,
            'response_format': { "type": "json_object" },
        },
    ]
else:
    config_list_json = [
        {
            'model': api_model,
            'api_key': api_key,
            'base_url': api_base,
            'api_version': api_version,
            'response_format': { "type": "json_object" },
        },
    ]


if api_type and api_type.startswith("azure"):

    config_list = [
        {
            'model': api_model,
            'api_key': api_key,
            'base_url': api_base,
            'api_type': api_type,
            'api_version': api_version,
        },
    ]
else:
    config_list = [
        {
            'model': api_model,
            'api_key': api_key,
            'base_url': api_base,
            'api_version': api_version,
        },
    ]

llm_config={
        "timeout": 600,
        "cache_seed": seed,
        "config_list": config_list,
        "temperature": 0,
        "top_p": 0,
    }

llm_config_json={
        "timeout": 600,
        "cache_seed": seed,
        "config_list": config_list_json,
        "temperature": 0,
        "top_p": 0,
    }


def init_TSG_Copilot():
    graph = nx.DiGraph()
    graph.add_node("user_proxy", label="user proxy")
    graph.add_node("node_retrieve_agent", label="node retrieve agent")
    graph.add_node("intent_understanding_agent", label="intent understanding agent")
    graph.add_node("planner_agent", label="planner agent")

    # Add edges between nodes
    graph.add_edge("user_proxy", "intent_understanding_agent")
    graph.add_edge("intent_understanding_agent", "user_proxy")
    graph.add_edge("intent_understanding_agent", "node_retrieve_agent")
    graph.add_edge("node_retrieve_agent", "intent_understanding_agent")
    graph.add_edge("intent_understanding_agent", "planner_agent")
    graph.add_edge("planner_agent", "user_proxy")

    # Set entry point
    graph.nodes["user_proxy"]["first_round_speaker"] = True
                

    # Termination message detection
    def is_termination_msg(content) -> bool:
        have_content = content.get("content", None) is not None
        if have_content and "TERMINATE" in content["content"]:
            return True
        return False

    def is_conversation_terminated(content) -> bool:
        have_content = content.get("content", None) is not None
        if have_content and content["content"].lower() == "exit":
            return True
        return False

    agents = []

    user_proxy = UserProxyAgent(
        name="user_proxy",
        human_input_mode="ALWAYS",
        system_message="An attentive HUMAN user who can answer questions about the task, and can perform tasks such as running Python code or inputting command line commands at a Linux terminal and reporting back the execution results.",
        code_execution_config=False,
        is_termination_msg=is_termination_msg,
        llm_config=llm_config
    )

    node_retrieve_agent = RetrieveAssistantAgent(
        name="node_retrieve_agent",
        human_input_mode="NEVER",
        # max_consecutive_auto_reply=3,
        retrieve_config={
            "tsg_path": tsg_path,
            "model": config_list[0]["model"],
            "embedding_function": openai_ef,
            "n_results": 5,
        },
        code_execution_config=False, # set to False if you don't want to execute the code
        llm_config=llm_config_json
    )

    intent_understanding_agent = IntentUnderstandingAgent(
        name="intent_understanding_agent",
        human_input_mode="NEVER",
        llm_config=llm_config_json,
    )

    planner_agent = PlannerAgent(
        name="planner_agent",
        human_input_mode="NEVER",
        llm_config=llm_config_json,
    )

    def print_messages(recipient, messages, sender, config):

        print(f"Messages from: {sender.name} sent to: {recipient.name} | num messages: {len(messages)} | message: {messages[-1]}")

        last_message=messages[-1]
        content=last_message['content']
        try:
            content=json.loads(content)
            if 'RESPONSE' in content.keys():
                try:
                    from termcolor import colored
                except ImportError:

                    def colored(x, *args, **kwargs):
                        return x
                
                to_print=content['RESPONSE']
                print(colored("==========", "red"), flush=True)
                print(colored(to_print, "red"), flush=True)
        except:
            return False, None
        return False, None

    user_proxy.register_reply(
        [autogen.Agent, None],
        reply_func=print_messages, 
        config={"callback": None},
    )

    node_retrieve_agent.register_reply(
        [autogen.Agent, None],
        reply_func=print_messages, 
        config={"callback": None},
    ) 

    intent_understanding_agent.register_reply(
        [autogen.Agent, None],
        reply_func=print_messages, 
        config={"callback": None},
    ) 

    planner_agent.register_reply(
        [autogen.Agent, None],
        reply_func=print_messages, 
        config={"callback": None},
    ) 

    agents.append(user_proxy)
    agents.append(node_retrieve_agent)
    agents.append(intent_understanding_agent)
    agents.append(planner_agent)

    group_chat = CustomGroupChat(
        agents=agents,  # Include all agents
        messages=[],
        max_round=max_round,
        graph=graph,
        l_user=l_user,
        l_exclude_assistant=l_exclude_assistant,
        l_oneway_assistant=l_oneway_assistant,
    )

    manager = GroupChatManager(
        name="chat_manager", 
        groupchat=group_chat, 
        llm_config=llm_config,
        is_termination_msg=is_conversation_terminated)

    state = TSG_Copilot_State(agents, manager)

    return state


# define state class that containing agents, group_chat, manager
class TSG_Copilot_State:
    def __init__(self, agents, manager):
        self.agents = agents        
        self.manager = manager

def TSG_Copilot_Chat(user_query, conversation_id, state, is_initial_conversation=False):
    agents=state.agents
    manager=state.manager
    try:
        if is_initial_conversation:
            agents[0].initiate_chat(manager, message=user_query)
        else:
            agents[0].continue_chat(manager, user_query=user_query)
    except OutputResultsError as output_results_error:
        prompt=output_results_error.message
        last_message = manager.groupchat.messages[-1]
        content=json.loads(last_message["content"])         
        content['prompt']=prompt
        if agents[1].previous_node != None:
            content['tsg'] = agents[1].previous_node["#title#"]
        else:
            content['tsg'] = ""
        return content
    except DeleteConversationError as delete_conversation_error:
        message=delete_conversation_error.message
        # delete the conversation
        assert conversation_id in copilot_state, "conversation_id not in copilot_state"
        del copilot_state[conversation_id]
        content={}
        content['prompt']='The conversation is over.'               
        content['RESPONSE']=message
        content['tsg'] = ""
        return content
    except MitigateConversationError as mitigate_conversation_error:
        prompt=mitigate_conversation_error.message
        last_message = manager.groupchat.messages[-1]
        content=json.loads(last_message["content"])         
        content['prompt']=prompt
        content['tsg'] = ""
        return content
    
def TSG_Copilot(user_input):
    user_query = user_input['query']
    conversation_id = user_input['conversation_id']
    if conversation_id not in copilot_state:
        # initialize the conversation
        state = init_TSG_Copilot()        
        results = TSG_Copilot_Chat(user_query, conversation_id, state, is_initial_conversation=True)        
    else:
        # load the conversation
        state = copilot_state[conversation_id]            
        assert state is not None, "state is None"
        results = TSG_Copilot_Chat(user_query, conversation_id, state)

    output={
        "prompt": results['prompt'],
        "response": results['RESPONSE'],
        "title": results['tsg'],
    }    
    # save state dict
    copilot_state[conversation_id] = state
    return output


@app.route('/api/tsg_copilot', methods=['POST'])
def copilot_handler():
    try:
        input_json = request.get_json(force=True)        
    except KeyError:
        return jsonify({'error': 'Invalid input JSON format. Required keys: sentences, chunks.'}), 400
    
    try:
        results = TSG_Copilot(input_json)        
    except IndexError:
        return jsonify({'error': 'Index out of range error.'}), 500 
    return jsonify(results)

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({'error': 'Internal Server Error.'}), 500


if __name__ == '__main__':
    app.run(debug=False, port=2000, host='0.0.0.0')
