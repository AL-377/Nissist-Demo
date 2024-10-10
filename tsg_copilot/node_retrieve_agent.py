import os
try:
    import chromadb
except ImportError:
    raise ImportError("Please install dependencies first. `pip install pyautogen[retrievechat]`")
from autogen.agentchat.agent import Agent
from .conversable_agent import ConversableAgent
from .node_retrieve_utils import create_vector_db_from_json_node, query_vector_db, TEXT_FORMATS
from autogen.token_count_utils import count_token
from autogen.code_utils import extract_code
from autogen import logger
import pickle
import json
from Kusto.kusto_api import query_kusto_api
from typing import Callable, Dict, Optional, Union, List, Tuple, Any

try:
    from termcolor import colored
except ImportError:

    def colored(x, *args, **kwargs):
        return x

SYSTEM_DEFAULT="""You are a helpful assistant that selects the most relevant element from a list of json formatted information in <INFO_LIST> based on the user's query in <USER_QUERY>. Please respond with the JSON format.
The each element in <INFO_LIST> are in json format and contains the following fields:
{
    "#type#": "the type of the element, select from the following types: terminology, background, FAQ, steps.",
    "#title#": "the description of the troubleshooting guide where this element comes from.",
    "#intent#": "the information describes the intent of the <INFO>.",
    "#action#": "the action is the content which troubleshoots the incident or give explanation of the #intent#. the action may contain code blocks in markdown format, and parameters are replaced with placeholders",
    "#output#": "the expected output after taking the #action#. It is defined in the following format in markdown: -If **condition**, then **should_do**. It can contain multiple if-then cases.",
    "#default_parameters#": "the default parameters that could fill in placeholders in code blocks in #output#."
}   
- The elements in <INFO_LIST> contains possible information that can answer the user's query in <USER_QUERY>. However, they may not be all relevant to the query or useful to answer the user's query. You should select the most relevant element from the <INFO_LIST> based on the user's query in <USER_QUERY>. 
- In particular, you should focus on the following fields in the element: #intent#, #action#. Most importantly, the <USER_QUERY> need to match with the #intent# and the #action# has to provide actions to reach the goal of the <USER_QUERY>, please ignore the #output# and do not map the <USER_QUERT> with #output#.
- Try to select only one element from <INFO_LIST>. If it is not possible to select only one element, you can select multiple elements from <INFO_LIST>.
- Your answer should be in the JSON format in a list after <RESPONSE>:
[
    {
        "INDEX": the index of the element in <INFO_LIST>.
        "INTENT": the #intent# of the element, the index starts from 0. 
        "EXPLANATION": justify why you select this node.
    }
]
- If there is no element in <INFO_LIST> that can answer the user's query in <USER_QUERY>, you should try select the most relevant element to the user's query considering that the user might use wrong terminology. And your answer should be in the JSON format in a list after <RESPONSE>:
[
    {
        "INDEX": the index of the element in <INFO_LIST>.
        "INTENT": the #intent# of the element, the index starts from 0. 
        "REPHRASED_QUERY": the rephrased query that you think the user is asking about.
        "EXPLANATION": justify why you select this node.
    }
]
- Unless you are confident that there is no element in <INFO_LIST> that is even close to the user's query, you should answer in the JSON format:
{
    "NO_INFO_EXPLANATION": where you give your explanation.
} 
"""


class RetrieveAssistantAgent(ConversableAgent):
    def __init__(
        self,
        name="RetrieveChatAgent",  # default set to RetrieveChatAgent
        human_input_mode: Optional[str] = "NEVER",
        is_termination_msg: Optional[Callable[[Dict], bool]] = None,
        retrieve_config: Optional[Dict] = None,  # config for the retrieve agent
        **kwargs,
    ):
        """
        Args:
            name (str): name of the agent.
            human_input_mode (str): whether to ask for human inputs every time a message is received.
                Possible values are "ALWAYS", "TERMINATE", "NEVER".
                (1) When "ALWAYS", the agent prompts for human input every time a message is received.
                    Under this mode, the conversation stops when the human input is "exit",
                    or when is_termination_msg is True and there is no human input.
                (2) When "TERMINATE", the agent only prompts for human input only when a termination message is received or
                    the number of auto reply reaches the max_consecutive_auto_reply.
                (3) When "NEVER", the agent will never prompt for human input. Under this mode, the conversation stops
                    when the number of auto reply reaches the max_consecutive_auto_reply or when is_termination_msg is True.
            is_termination_msg (function): a function that takes a message in the form of a dictionary
                and returns a boolean value indicating if this received message is a termination message.
                The dict can contain the following keys: "content", "role", "name", "function_call".
            retrieve_config (dict or None): config for the retrieve agent.
                To use default config, set to None. Otherwise, set to a dictionary with the following keys:
                - task (Optional, str): the task of the retrieve chat. Possible values are "code", "qa" and "default". System
                    prompt will be different for different tasks. The default value is `default`, which supports both code and qa.
                - client (Optional, chromadb.Client): the chromadb client. If key not provided, a default client `chromadb.Client()`
                    will be used. If you want to use other vector db, extend this class and override the `retrieve_docs` function.
                - tsg_path (Optional, Union[str, List[str]]): the path to the nodedb directory. It can also be the path to a single file,
                    the url to a single file or a list of directories, files and urls. Default is None, which works only if the collection is already created.
                - collection_name (Optional, str): the name of the collection.
                    If key not provided, a default name `autogen-docs` will be used.
                - model (Optional, str): the model to use for the retrieve chat.
                    If key not provided, a default model `gpt-4` will be used.
                - chunk_token_size (Optional, int): the chunk token size for the retrieve chat.
                    If key not provided, a default size `max_tokens * 0.4` will be used.
                - context_max_tokens (Optional, int): the context max token size for the retrieve chat.
                    If key not provided, a default size `max_tokens * 0.8` will be used.
                - chunk_mode (Optional, str): the chunk mode for the retrieve chat. Possible values are
                    "multi_lines" and "one_line". If key not provided, a default mode `multi_lines` will be used.
                - must_break_at_empty_line (Optional, bool): chunk will only break at empty line if True. Default is True.
                    If chunk_mode is "one_line", this parameter will be ignored.
                - embedding_model (Optional, str): the embedding model to use for the retrieve chat.
                    If key not provided, a default model `all-MiniLM-L6-v2` will be used. All available models
                    can be found at `https://www.sbert.net/docs/pretrained_models.html`. The default model is a
                    fast model. If you want to use a high performance model, `all-mpnet-base-v2` is recommended.
                - embedding_function (Optional, Callable): the embedding function for creating the vector db. Default is None,
                    SentenceTransformer with the given `embedding_model` will be used. If you want to use OpenAI, Cohere, HuggingFace or
                    other embedding functions, you can pass it here, follow the examples in `https://docs.trychroma.com/embeddings`.
                - customized_prompt (Optional, str): the customized prompt for the retrieve chat. Default is None.
                - customized_answer_prefix (Optional, str): the customized answer prefix for the retrieve chat. Default is "".
                    If not "" and the customized_answer_prefix is not in the answer, `Update Context` will be triggered.
                - update_context (Optional, bool): if False, will not apply `Update Context` for interactive retrieval. Default is True.
                - get_or_create (Optional, bool): if True, will create/return a collection for the retrieve chat. This is the same as that used in chromadb.
                    Default is False. Will raise ValueError if the collection already exists and get_or_create is False. Will be set to True if tsg_path is None.
                - custom_token_count_function (Optional, Callable): a custom function to count the number of tokens in a string.
                    The function should take (text:str, model:str) as input and return the token_count(int). the retrieve_config["model"] will be passed in the function.
                    Default is autogen.token_count_utils.count_token that uses tiktoken, which may not be accurate for non-OpenAI models.
                - custom_text_split_function (Optional, Callable): a custom function to split a string into a list of strings.
                    Default is None, will use the default function in `autogen.retrieve_utils.split_text_to_chunks`.
                - custom_text_types (Optional, List[str]): a list of file types to be processed. Default is `autogen.retrieve_utils.TEXT_FORMATS`.
                    This only applies to files under the directories in `tsg_path`. Explictly included files and urls will be chunked regardless of their types.
                - recursive (Optional, bool): whether to search documents recursively in the tsg_path. Default is True.
            **kwargs (dict): other kwargs in [UserProxyAgent](../user_proxy_agent#__init__).

        Example of overriding retrieve_docs:
        If you have set up a customized vector db, and it's not compatible with chromadb, you can easily plug in it with below code.
        ```python
        class MyRetrieveUserProxyAgent(RetrieveUserProxyAgent):
            def query_vector_db(
                self,
                query_texts: List[str],
                n_results: int = 10,
                search_string: str = "",
                **kwargs,
            ) -> Dict[str, Union[List[str], List[List[str]]]]:
                # define your own query function here
                pass

            def retrieve_docs(self, problem: str, n_results: int = 20, search_string: str = "", **kwargs):
                results = self.query_vector_db(
                    query_texts=[problem],
                    n_results=n_results,
                    search_string=search_string,
                    **kwargs,
                )

                self._results = results
                print("doc_ids: ", results["ids"])
        ```
        """
        super().__init__(
            name=name,
            human_input_mode=human_input_mode,
            **kwargs,
        )

        self._retrieve_config = {} if retrieve_config is None else retrieve_config
        # self._task = self._retrieve_config.get("task", "default")
        self._client = self._retrieve_config.get("client", chromadb.Client())
        self._tsg_path = self._retrieve_config.get("tsg_path", None)
        if "tsg_path" not in self._retrieve_config:
            logger.warning(
                "tsg_path is not provided in retrieve_config. "
                f"Will raise ValueError if the collection `{self._collection_name}` doesn't exist. "
                "Set tsg_path to None to suppress this warning."
            )                
        self._collection_name = self._retrieve_config.get("collection_name", "all-tsg-nodes")
        self._n_results= self._retrieve_config.get("n_results", 10)
        self._model = self._retrieve_config.get("model", "gpt-4")
        self._update_db = self._retrieve_config.get("update_db", False)
        # self._create_db = self._retrieve_config.get("create_db", False)
        # self._max_tokens = self.get_max_tokens(self._model)
        # self._chunk_token_size = int(self._retrieve_config.get("chunk_token_size", self._max_tokens * 0.4))
        # self._chunk_mode = self._retrieve_config.get("chunk_mode", "multi_lines")
        # self._must_break_at_empty_line = self._retrieve_config.get("must_break_at_empty_line", True)
        self._embedding_model = self._retrieve_config.get("embedding_model", "all-MiniLM-L6-v2")
        self._embedding_function = self._retrieve_config.get("embedding_function", None)
        # self.customized_prompt = self._retrieve_config.get("customized_prompt", None)
        # self.customized_answer_prefix = self._retrieve_config.get("customized_answer_prefix", "").upper()
        # self.update_context = self._retrieve_config.get("update_context", True)
        # self._get_or_create = self._retrieve_config.get("get_or_create", False) if self._tsg_path is not None else True
        # self.custom_token_count_function = self._retrieve_config.get("custom_token_count_function", count_token)
        # self.custom_text_split_function = self._retrieve_config.get("custom_text_split_function", None)
        # self._custom_text_types = self._retrieve_config.get("custom_text_types", TEXT_FORMATS)
        # self._recursive = self._retrieve_config.get("recursive", True)
        # self._context_max_tokens = self._max_tokens * 0.8
        # self._collection = True if self._tsg_path is None else False  # whether the collection is created
        # self._ipython = get_ipython()
        self._parent_path = os.path.dirname(self._tsg_path)
        self._kv_map_path = os.path.join(self._parent_path, "kv_map.pkl")
        self._nodedb_path = os.path.join(self._parent_path, "nodedb")
        self._doc_idx = -1  # the index of the current used doc
        self._results = {}  # the results of the current query
        # self._intermediate_answers = set()  # the intermediate answers
        self._doc_contents = []  # the contents of the current used doc
        self._doc_ids = []  # the ids of the current used doc
        self._search_string = ""  # the search string used in the current query
        self._kv_map = []  # the kv map of tsg nodes
        self._marker = None  # the marker of the kv map
        self.previous_node = None  # store the previous step's retrievd node
        # update the termination message function
        self._is_termination_msg = (
            self._is_termination_msg_retrievechat if is_termination_msg is None else is_termination_msg
        )
        self._create_node_db()
        self.register_reply(Agent, RetrieveAssistantAgent._generate_retrieve_user_reply, position=2)

    def _check_nodedb_exists(self):
        if not os.path.exists(self._kv_map_path) or not os.path.exists(self._nodedb_path):
            return False
        files = [f for f in os.listdir(self._nodedb_path) if os.path.isfile(os.path.join(self._nodedb_path, f))]
        if not files:
            return False
        return True

    def _create_node_db(self):        
        # load the kv map
        # if not self._collection or not self._get_or_create:
        if not self._check_nodedb_exists():
            print("Trying to create collection.")
            self._client, self._kv_map = create_vector_db_from_json_node(
                tsg_path=self._tsg_path,
                kv_map_path=self._kv_map_path,
                db_path=self._nodedb_path,
                # client=self._client,
                collection_name=self._collection_name,                
                embedding_model=self._embedding_model,
                embedding_function=self._embedding_function,                
            )
        else:
            self._load_kv_map()

    def _load_kv_map(self):
        assert self._tsg_path is not None, "tsg_path is not provided."
        # get the parent directory of self._tsg_path
        with open(self._kv_map_path, "rb") as f:
            self._kv_map, self._marker = pickle.load(f)

    def _is_termination_msg_retrievechat(self, message):
        """Check if a message is a termination message.
        For code generation, terminate when no code block is detected. Currently only detect python code blocks.
        For question answering, terminate when don't update context, i.e., answer is given.
        """
        if isinstance(message, dict):
            message = message.get("content")
            if message is None:
                return False
        
        return False

    @staticmethod
    def get_max_tokens(model="gpt-3.5-turbo"):
        if "32k" in model:
            return 32000
        elif "16k" in model:
            return 16000
        elif "gpt-4" in model:
            return 8000
        else:
            return 4000

    def _reset(self, intermediate=False):
        self._doc_idx = -1  # the index of the current used doc
        self._results = {}  # the results of the current query
        if not intermediate:
            self._intermediate_answers = set()  # the intermediate answers
            self._doc_contents = []  # the contents of the current used doc
            self._doc_ids = []  # the ids of the current used doc

    def _get_node_json_list(self, results: Dict[str, Union[List[str], List[List[str]]]]):
        l_node_json=[]
        for id in results['ids'][0]:
            # get id from 'node_id'
            # print("id", id, len(self._kv_map))
            id = id.split('_')[1]
            if int(id) < 0 or int(id) > len(self._kv_map):
                continue
            l_node_json.append(self._kv_map[int(id)])
        return l_node_json

    def _generate_message(self, l_node_json):
        if not l_node_json:
            print(colored("No more context, will terminate.", "green"), flush=True)
            return "TERMINATE"
        message_list = [
            {
                "role": "system",
                "content": SYSTEM_DEFAULT
            },
            {
                "role": "user",
                "content": "Here is the user's query and information list:\n<USER_QUERY>:\n{user_query}\n<INFO_LIST>:\n{l_node_json}\n<RESPONSE>:\n".format(user_query=self.problem, l_node_json=l_node_json)
            }
        ]
        for node_json in l_node_json:
            print(node_json)
            print("=============")
        message = self.generate_oai_reply_self(message_list=message_list)

        return message

    def retrieve_docs(self, problem: str, n_results: int = 10, search_string: str = "", where: dict = None):
        """Retrieve docs based on the given problem and assign the results to the class property `_results`.
        In case you want to customize the retrieval process, such as using a different vector db whose APIs are not
        compatible with chromadb or filter results with metadata, you can override this function. Just keep the current
        parameters and add your own parameters with default values, and keep the results in below type.

        Type of the results: Dict[str, List[List[Any]]], should have keys "ids" and "documents", "ids" for the ids of
        the retrieved docs and "documents" for the contents of the retrieved docs. Any other keys are optional. Refer
        to `chromadb.api.types.QueryResult` as an example.
            ids: List[string]
            documents: List[List[string]]

        Args:
            problem (str): the problem to be solved.
            n_results (int): the number of results to be retrieved. Default is 20.
            search_string (str): only docs that contain an exact match of this string will be retrieved. Default is "".
        """
        # print(f"Search String:\n{search_string}")
        # print(f"Problem:\n{problem}")
        results = query_vector_db(
            query_texts=[problem],
            n_results=n_results,
            search_string=search_string,
            # client=self._client,
            db_path=self._nodedb_path,
            collection_name=self._collection_name,
            embedding_model=self._embedding_model,
            embedding_function=self._embedding_function,
            where=where
        )
        
        self._search_string = search_string
        self._results = results


    def generate_init_message(self, problem: str, search_string: str = ""):
        """Generate an initial message with the given problem and prompt.

        Args:
            problem (str): the problem to be solved.
            n_results (int): the number of results to be retrieved.
            search_string (str): only docs containing this string will be retrieved.

        Returns:
            str: the generated prompt ready to be sent to the assistant agent.
        """
        self._reset()
        
        self.retrieve_docs(problem, self._n_results, search_string)
        self.problem = problem
        l_node_json = self._get_node_json_list(self._results)
        message = self._generate_message(l_node_json)
        return message
    
    def _generate_retrieve_user_reply(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[Any] = None,
    ) -> Tuple[bool, Union[str, Dict, None]]:
        """In this function, we will update the context and reset the conversation based on different conditions.
        We'll update the context and reset the conversation if update_context is True and either of the following:
        (1) the last message contains "UPDATE CONTEXT",
        (2) the last message doesn't contain "UPDATE CONTEXT" and the customized_answer_prefix is not in the message.
        """
        if config is None:
            config = self
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        content = json.loads(message['content'])

        if 'QUERY' in content:
            problem = content['QUERY']
            special_token = content['TOKEN']
            self._reset(intermediate=True)
            # add filter condition: monitorId 
            self.retrieve_docs(problem, self._n_results)
            self.problem=problem
            l_node_json = self._get_node_json_list(self._results)
            message = self._generate_message(l_node_json)
            message = json.loads(message)
            if isinstance(message, list):
                message = message[0]

            if 'INDEX' in message and 'EXPLANATION' in message:
                index = message['INDEX']
                l_selected_node_json = l_node_json[index]
                explanation = message['EXPLANATION']
                # add to judge if newly retrieved one is same with the latest in the <chat_history>
                if sender.name == 'chat_manager':
                    if self.previous_node is not None:  # to check if the retrieve node reasonable 
                        current_node = l_selected_node_json
                        print("selected node", current_node)
                        if special_token == "[CONTINUE]" or "CONTINUE" in special_token:  # as normal
                            # the node should be different but with same tsg title
                            if not (current_node["#title#"] == self.previous_node["#title#"] and \
                                current_node != self.previous_node):
                                new_message={}
                                new_message['no_info']="The newly retrieved node can not help solve your input with the last plan, please feel free to get connect with oncall engnieers."
                                new_message=json.dumps(new_message)
                                message['content'] = new_message
                                self.previous_node = None
                                return True, message
                        elif special_token == "[CROSS]" or "CROSS" in special_token:
                            # the node should be different and with different tsg title
                            if not (current_node["#title#"] != self.previous_node["#title#"] and \
                                current_node != self.previous_node):          
                                self.previous_node = l_selected_node_json
                                new_message={}
                                new_message['no_info']="Your input triggers a new step which does not out of the knowledge of Copilot, please feel free to get connect with oncall engnieers."
                                new_message=json.dumps(new_message)
                                message['content'] = new_message
                                self.previous_node = None
                                return True, message
                        elif special_token == "[MITIGATE]" or "MITIGATE" in special_token:
                            # actually, it should not retrieve any nodes, but it does not matter
                            new_message={}
                            new_message['no_info']="The incident is mitigated successfully, feel free to ask more questions."
                            new_message=json.dumps(new_message)
                            message['content'] = new_message
                            return True, message
                if 'REPHRASED_QUERY' in message:
                    rephrased_query = message['REPHRASED_QUERY']
                    explanation = explanation + "\n" + "Rephrased query: " + rephrased_query
                new_message={}
                new_message['info']=l_selected_node_json
                new_message['content']=explanation
                new_message['query']=problem
                # convert dict to str
                new_message=json.dumps(new_message)
                message['content']=new_message
                self.clear_history()
                self.previous_node = l_selected_node_json      # update the current step's retrieved node
                # sender.clear_history()
                return True, message
            elif "NO_INFO_EXPLANATION" in message:      # no retrieval information, maybe the query is too specific
                if special_token == "[MITIGATE]" or "MITIGATE" in special_token:
                    # actually, it should not retrieve any nodes, but it does not matter
                    new_message={}
                    new_message['no_info']="The incident is mitigated successfully, feel free to ask more questions."
                    new_message=json.dumps(new_message)
                    message['content'] = new_message
                    self.previous_node = None
                    return True, message
                self.previous_node = None
                new_message={}
                new_message['no_info']=message['NO_INFO_EXPLANATION']
                new_message=json.dumps(new_message)
                message['content'] = new_message
                return True, json.dumps(message)
            else:
                return False, None
        # Add if user input IncidentId
        elif "IncidentId" in content:
            incidentid = content["IncidentId"]
            query = """
            cluster('icmclustermirrorsre.kusto.windows.net').database("IcMDataWarehouse").Incidents
            | where  IncidentId == {}
            | summarize arg_max(Lens_IngestionTime, *) by IncidentId
            """.format(incidentid)
            info = query_kusto_api(query)
            monitorid = info["MonitorId"].values[0]
            title = info["Title"].values[0]
            summary = info["Summary"].values[0]
            start = info["ImpactStartDate"].values[0]
            end = info["MitigateDate"].values[0]
            self._reset(intermediate=True)
            self.problem = query
            where={
                "$and": [
                    {"monitor": monitorid},
                    {"isfirst": True}
                ]
            }
            self.retrieve_docs(self.problem, self._n_results, where=where)
            l_node_json = self._get_node_json_list(self._results)

            new_message={}
            content = f"There is the incident details: **Incident**: {title}\n **Starttime**: {start}\n **Endtime**: {end}\n **Summary**: {summary}"
            new_message['info'] = l_node_json[0]
            new_message['info']['#incident_details#']=content
            new_message['query'] = self.problem
            message = {'content': json.dumps(new_message)}
            self.previous_node = l_node_json[0]
            return True, message
        else:
            return False, None


    def run_code(self, code, **kwargs):
        lang = kwargs.get("lang", None)
        if code.startswith("!") or code.startswith("pip") or lang in ["bash", "shell", "sh"]:
            return (
                0,
                "You MUST NOT install any packages because all the packages needed are already installed.",
                None,
            )
        if self._ipython is None or lang != "python":
            return super().run_code(code, **kwargs)
        else:
            result = self._ipython.run_cell(code)
            log = str(result.result)
            exitcode = 0 if result.success else 1
            if result.error_before_exec is not None:
                log += f"\n{result.error_before_exec}"
                exitcode = 1
            if result.error_in_exec is not None:
                log += f"\n{result.error_in_exec}"
                exitcode = 1
            return exitcode, log, None
