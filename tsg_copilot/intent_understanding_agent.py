from typing import Callable, Dict, Literal, Optional, Union

from .conversable_agent import ConversableAgent
from autogen import Agent
import json

DEFAULT_SYSTEM_MESSAGE = """You are a helpful troubleshooting guide Copilot, i.e., TSG Copilot, that helps the user to troubleshoot the user's query in <USER_QUERY>, with chat history between TSG Copilot and user in <CHAT_HISTORY> and <INFO> retrieved from the knowledge base that helps to make plan for <USER_QUERY>.

"""

NEW_QUERY_MESSAGE = """You should judge that if <USER_QUERY> expresses their intent to troubleshoot incidents by your own knowledge. 
- If the <USER_QUERY> gives an IncidentId that user just encounter, then you should give it to node_retrieve_agent to retrieve information about the incident. Your answer should be in the json format after <RESPONSE>:
{
    "DECISION": "NEW_INCIDENT",
    "NEXT": "node_retrieve_agent",
    "IncidentId": the IncidentId from <USER_QUERY>,
    "TOKEN": "[CONTINUE]",
    "RESPONSE": your response here.
}
- Else, if it is relevant to incidents, you should ask node_retrieve_agent to retrieve the information from the TSGs. Your answer should be in the json format after <RESPONSE>:
{
    "DECISION": "NEW_QUERY",
    "NEXT": "node_retrieve_agent",
    "QUERY": the SAME query from <USER_QUERY>,
    "TOKEN": "[CONTINUE]",
    "RESPONSE": your response here.
}
- Else, if the <USER_QUERY> is not related to any incident troubleshooting guide (TSG), you should clarify that you are a TSG Copilot and suggest the user to ask questions on incident troubleshooting. Your answer should be in the json format after <RESPONSE>:
{
    "DECISION": "NOT_RELATED",
    "NEXT": "user_proxy",
    "RESPONSE": put your response on <USER_QUERY> here.
}
"""

JUDGE_QUERT_MESSAGE = """- You should judge if <USER_QUERY> is the user's response to the latest plan in <CHAT_HISTORY>.
    - If <USER_QUERY> directly asks like "what to do next" or similar queries, it means the user is moving on with latest plan within <CHAT_HISTORY> or that <USER_QUERY> provides some results or descriptions and matches with the "if" conditions or other action triggers in the latest plan in <CHAT_HISTORY>, please utilize the corresponding "then" outcomes and request the node_retrieve_agent to retrieve the necessary information. Ensure your response is formatted in JSON following <RESPONSE>. If the plan's expected output is a more descriptive analysis rather than a direct "if-then" structure, use your judgement to extract the most relevant information to address the user's query. (Note that this branch, the <INFO> is empty.)
    {
        "DECISION": "MATCHED",
        "NEXT": "node_retrieve_agent",
        "QUERY": the corresponding "then" outcomes or actions need to take in the latest plan in <CHAT_HISTORY>, not including the the special token,
        "TOKEN": the special token behind the corresponding action, should be "[MITIGATE]", "[CROSS]" or "[CONTINUE]",
        "RESPONSE": your response here.
    }
    - If not matches but <USER_QUERY> expresses their intent to troubleshoot, it means that <USER_QUERY> is not related to previous history, you should ask node_retrieve_agent to retrieve the information. Your answer should be in the json format after <RESPONSE>:
    {
        "DECISION": "NEW_QUERY",
        "NEXT": "node_retrieve_agent",
        "QUERY": the SAME query from <USER_QUERY>,
        "TOKEN": "[CONTINUE]",
        "RESPONSE": new query with your response here.
    } 
    - If <USER_QUERY> provides the required parameters of the latest plan in <CHAT_HISTORY> such as cluster or container, or implies current plan is not good enough, you should ask planner_agent to add the parameters into the plan or refine it. Your answer should be in the json format after <RESPONSE>:
    {
        "DECISION": "REFINE",
        "NEXT": "planner_agent",
        "RESPONSE": your requirment that ask planner_agent to incorporate parameters or refine the plans.
    }
"""

JUDGE_INFO_MESSAGE = """- You should check if <INFO> contains necessary information to answer <USER_QUERY>. 
    - If <INFO> contains necessary information, you should ask planner_agent to plan the steps to troubleshoot the incident and gives <INFO> to planner_agent. Your answer should be in the json format after <RESPONSE>:
    {
        "DECISION": "RELATED",
        "NEXT": "planner_agent",
        "QUERY": the <USER_QUERY> received,
        "RESPONSE": the SAME information from <INFO>, NOT from Node Retrieval.
    }
    - If <INFO> does NOT contain enough or necessary information, it might be because <USER_QUERY> is not clarified enough. You should rephrase <USER_QUERY> based on <INFO> and ask if the user is asking about the rephrased query. Your answer should be in the json format after <RESPONSE>:
    {
        "DECISION": "RELATED",
        "NEXT": "user_proxy",
        "RESPONSE": your rephrased query.
    }
"""

IRRELEVANT_MESSAGE = """- If <INFO> expresses that the retrieved <INFO_LIST> does not include any information directly related to the user's query, including some information about "NO_INFO_EXPLANATION", or that the <USER_QUERY> is not related to any incident troubleshooting guide (TSG), you should clarify that you are a TSG Copilot and suggest the user to ask questions on incident troubleshooting. Your answer should be in the json format after <RESPONSE>:
{
    "DECISION": "NOT_RELATED",
    "NEXT": "user_proxy",
    "RESPONSE": put your response on <USER_QUERY> here.
}
- If <USER_QUERY> expresses that 'please feel free to get connect with oncall engnieers'or that the incident has been mitigated, you should return to user and tell the user this incident is mitigated or should ask oncall engineers for some help. Your answert should be in the json format after <RESPONSE>:
{
    "DECISION": "MITIGATED",
    "NEXT": "user_proxy",
    "RESPONSE": your response about this incident is mitigated.
}
"""

DEFAULT_RECEIVE_MESSAGE="""<USER_QUERY>:
{user_query}
<INFO>:
{info}
<CHAT_HISTORY>:
{chat_history}
<RESPONSE>:
"""

DEFAULT_DESCRIPTION = "A intent understanding agent of TSG Copilot."




class IntentUnderstandingAgent(ConversableAgent):
    """(In preview) Assistant agent, designed to solve a task with LLM.

    AssistantAgent is a subclass of ConversableAgent configured with a default system message.
    The default system message is designed to solve a task with LLM,
    including suggesting python code blocks and debugging.
    `human_input_mode` is default to "NEVER"
    and `code_execution_config` is default to False.
    This agent doesn't execute code by default, and expects the user to execute the code.
    """

    def __init__(
        self,
        name: str,
        system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
        llm_config: Optional[Union[Dict, Literal[False]]] = None,
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        max_consecutive_auto_reply: Optional[int] = None,
        human_input_mode: Optional[str] = "NEVER",
        code_execution_config: Optional[Union[Dict, Literal[False]]] = False,
        description: Optional[str] = None,
        **kwargs,
    ):
        """
        Args:
            name (str): agent name.
            system_message (str): system message for the ChatCompletion inference.
                Please override this attribute if you want to reprogram the agent.
            llm_config (dict): llm inference configuration.
                Please refer to [OpenAIWrapper.create](/docs/reference/oai/client#create)
                for available options.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            max_consecutive_auto_reply (int): the maximum number of consecutive auto replies.
                default to None (no limit provided, class attribute MAX_CONSECUTIVE_AUTO_REPLY will be used as the limit in this case).
                The limit only plays a role when human_input_mode is not "ALWAYS".
            **kwargs (dict): Please refer to other kwargs in
                [ConversableAgent](conversable_agent#__init__).
        """
        super().__init__(
            name,
            system_message,
            is_termination_msg,
            max_consecutive_auto_reply,
            human_input_mode,
            code_execution_config=code_execution_config,
            llm_config=llm_config,
            description=description,
            **kwargs,
        )

        # Update the provided desciption if None, and we are using the default system_message,
        # then use the default description.
        if description is None:
            if system_message == DEFAULT_SYSTEM_MESSAGE:
                self.description = DEFAULT_DESCRIPTION

    def _process_received_message(self, message: Union[Dict, str], sender: Agent, silent: bool):
        message = self._message_to_dict(message)

        # update context with the received message
        update_content=message['content']
        print("intent received", update_content)

        info=""
        query=""
        no_info = ""

        try:
            update_content=json.loads(update_content)
            query = update_content.get('query', '')
            info = update_content.get('info', '')
            no_info = update_content.get('no_info', '')
        except:
            print('intent agent: failed in converting to json')
            query=update_content

        # get chat history from chat manager
        if sender.name == 'chat_manager':
            memory = sender.get_memory()
            if memory:
                chat_history = '\n'.join(conversation['chat'] for conversation in memory)
            else:
                chat_history = ''

        if info == "" and no_info == "" and len(memory) == 1:  # initial call
            self._oai_system_message = [{"content": DEFAULT_SYSTEM_MESSAGE + NEW_QUERY_MESSAGE, "role": "system"}]
            message = DEFAULT_RECEIVE_MESSAGE.format(user_query=query, info=info, chat_history=chat_history)
        elif info == "" and no_info != "":  # no relevant node is retrieved
            self._oai_system_message = [{"content": DEFAULT_SYSTEM_MESSAGE + IRRELEVANT_MESSAGE, "role": "system"}]
            message = DEFAULT_RECEIVE_MESSAGE.format(user_query=query, info=no_info, chat_history=chat_history)
        elif info == "" and chat_history != "": # get info from user response
            self._oai_system_message = [{"content": DEFAULT_SYSTEM_MESSAGE + JUDGE_QUERT_MESSAGE, "role": "system"}]
            message = DEFAULT_RECEIVE_MESSAGE.format(user_query=query, info=info, chat_history=chat_history)
        elif info != "":    # get info from node retriever
            self._oai_system_message = [{"content": DEFAULT_SYSTEM_MESSAGE + JUDGE_INFO_MESSAGE, "role": "system"}]
            message = DEFAULT_RECEIVE_MESSAGE.format(user_query=query, info=info, chat_history=chat_history)

        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_oai_message(message, "user", sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )
        if not silent:
            self._print_received_message(message, sender)