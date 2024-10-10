from typing import Callable, Dict, Literal, Optional, Union

from .conversable_agent import ConversableAgent
from autogen import Agent
import json


DEFAULT_SYSTEM_MESSAGE = """You are a helpful troubleshooting guide assistant that helps the user to troubleshoot the incident or the query in <USER_QUERY>. You will be provided with information related to the user's incident in <INFO>. <INFO> is in json format and contains the following fields: 
{
    "#type#": "the type of the element, select from the following types: terminology, background, FAQ, steps.",
    "#title#": "the description of the troubleshooting guide where this element comes from.",
    "#intent#": "the information describes the intent of the <INFO>.",
    "#action#": "the action is the content which troubleshoots the incident or give explanation of the #intent#. the action may contain code template, which are code blocks in markdown format, where parameters are replaced with placeholders",
    "#output#": "the expected output after taking the #action#. It is defined in the following format in markdown: -If **condition**, then **should_do**. It can contain multiple if-then cases.",
    "#default_parameters#": "the default parameters that could fill in placeholders in code blocks in #output#."
}   
- You will be provided with chat history in <CHAT_HISTORY> where contains the previous conversation between the user and the chatbot. You can use the chat history to generate the answer and parameters provided by the user.
- If <INFO> is empty, it means that you should refine you latest plans in <CHAT_HISTORY> with <USER_QUERY> and using the latest Retrieved Node as <INFO> in <CHAT_HISTORY>, either add the specific parameters or polish the current plans.
- You should answer the <USER_QUERY> based on the information in <INFO> and <CHAT_HISTORY>. 
- <CHAT_HISTORY> can be empty if tue <USER_QUERY> is the first query or not related to the previous conversation.
- Your answer should fullfill the following requirements:
    - Do NOT simply answer something like 'follow the provided steps', 'run the query'. 
    - Your answer should be specific and detailed including the action instructions and code blocks from <INFO>. 
    - The information in #action# part of <INFO> should be fully preserved.
    - If there is any expected output, you should also provide the expected output after taking the action following the #output# part. 
    - Your answer should be in the json format after <RESPONSE>:
    {
        "RESPONSE": where you provide the actions to troubleshooting the incident or the <USER_QUERY>. 
    }
- If user provides the incident information with parameters that can be used to fill in the placeholders in the code template, please fill in the placeholders with the parameters provided by the user.
- If the provided parameters are not enough to fill in the placeholders in the code template, please notify the user to provide the missing parameter, and in the meanwhile, fill in the placeholder with default parameters in #default_parameters#.
- You should give complete code, not using "Rest of the query remains unchanged" and so on to omit.
- If user provides the incident information, but did not include any parameters can be used to fill in the placeholders in the code template, please notify the user to provide the missing information, and in the meanwhile, do not forget to fill in the placeholder with default parameters in #default_parameters#.
- If <CHAT_HISTORY> contains the user's parameter, you should use it to replace the placeholder.
- You should not include any special tags which are stated in the prompt in your answer, such as <INFO>, <CHAT_HISTORY>, <USER_QUERY>.
- You should not use your own knowledge in the answer, only use information from <INFO> and <CHAT_HISTORY>.
Below is the <INFO> and <CHAT_HISTORY>:
"""

DEFAULT_RECEIVE_MESSAGE="""<USER_QUERY>:
{user_query}
<INFO>:
{info}
<CHAT_HISTORY>:
{chat_history}
<RESPONSE>:
"""

DEFAULT_DESCRIPTION = "A helpful and general-purpose AI assistant that can make incident mitigation plans."


class PlannerAgent(ConversableAgent):
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
        info=""
        query=""

        try:
            update_content=json.loads(update_content)
            decision = update_content['DECISION']
            if decision == "RELATED":
                if "QUERY" in update_content.keys():
                    query=update_content['QUERY']
                if "RESPONSE" in update_content.keys():
                    # use offline node avoiding LLMs rewrite the node info
                    info = sender.groupchat.agents[1].previous_node
            elif decision == "REFINE":
                query=update_content['RESPONSE']
            else:
                query=update_content['QUERY']
                info=update_content['RESPONSE']
        except:
            print('planner agent: failed in converting to json')
            query=update_content

        # get chat history from chat manager
        if sender.name == 'chat_manager':
            memory = sender.get_memory()
            if memory:
                chat_history = '\n'.join(conversation['chat'] for conversation in memory)
            else:
                chat_history = ''

        message = DEFAULT_RECEIVE_MESSAGE.format(user_query=query, info=info, chat_history=chat_history)

        # When the agent receives a message, the role of the message is "user". (If 'role' exists and is 'function', it will remain unchanged.)
        valid = self._append_oai_message(message, "user", sender)
        if not valid:
            raise ValueError(
                "Received message can't be converted into a valid ChatCompletion message. Either content or function_call must be provided."
            )
        if not silent:
            self._print_received_message(message, sender)