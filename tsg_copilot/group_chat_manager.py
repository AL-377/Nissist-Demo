import logging
import random
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union, Tuple

from autogen.code_utils import content_str
from autogen import Agent
from tsg_copilot.conversable_agent import ConversableAgent
from tsg_copilot.group_chat import CustomGroupChat
from tsg_copilot.utils import DeleteConversationError, MitigateConversationError

logger = logging.getLogger(__name__)


class GroupChatManager(ConversableAgent):
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: CustomGroupChat,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[Union[str, List]] = "Group chat manager.",
        **kwargs,
    ):
        if kwargs.get("llm_config") and (kwargs["llm_config"].get("functions") or kwargs["llm_config"].get("tools")):
            raise ValueError(
                "GroupChatManager is not allowed to make function/tool calls. Please remove the 'functions' or 'tools' config in 'llm_config' you passed in."
            )

        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )
        # Order of register_reply is important.
        # Allow sync chat if initiated using initiate_chat
        self.register_reply(Agent, GroupChatManager.run_chat, config=groupchat, reset_config=CustomGroupChat.reset)
        # Allow async chat if initiated using a_initiate_chat
        self.register_reply(Agent, GroupChatManager.a_run_chat, config=groupchat, reset_config=CustomGroupChat.reset)
        self.groupchat = groupchat
        self.chat_round = 0
    
    def get_memory(self) -> List[Dict]:
        """Return the memory of the agent."""
        memory=self.groupchat.memory
        return memory

    def run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[CustomGroupChat] = None,
    ) -> Union[str, Dict, None]:
        """Run a group chat."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        group_max_round = groupchat.max_round-self.chat_round
        for i in range(group_max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name

            groupchat.messages.append(message)

            if self._is_termination_msg(message):
                # The conversation is over
                raise DeleteConversationError()
                break

            # add to check if intent return with [MITIGATE] special token
            if speaker.name == "intent_understanding_agent":
                import json
                temp_intent = json.loads(message["content"])
                try:
                    special_token = temp_intent["TOKEN"]
                    if special_token == "[MITIGATE]" or "MITIGATE" in special_token:
                        raise MitigateConversationError()
                except MitigateConversationError:  
                    raise MitigateConversationError()
                except:
                    pass
            if i == group_max_round - 1:
                raise DeleteConversationError()
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                self.send(message, speaker, request_reply=False, silent=True)
                # let the speaker speak
                reply = speaker.generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            speaker.send(reply, self, request_reply=False)
            message = self.last_message(speaker)

        return True, None

    async def a_run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Agent] = None,
        config: Optional[CustomGroupChat] = None,
    ):
        """Run a group chat asynchronously."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name

            groupchat.messages.append(message)

            if self._is_termination_msg(message):
                # The conversation is over
                break
            
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = await groupchat.a_select_speaker(speaker, self)
                # let the speaker speak
                self.send(message, speaker, request_reply=False, silent=True)
                reply = await speaker.a_generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = await speaker.a_generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            await speaker.a_send(reply, self, request_reply=False)
            message = self.last_message(speaker)
        return True, None
