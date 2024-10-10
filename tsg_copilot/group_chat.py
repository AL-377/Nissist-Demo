from autogen.agentchat.groupchat import GroupChat
import random
import json

class CustomGroupChat(GroupChat):
    def __init__(self, agents, messages, max_round=10, graph=None, l_user=[], l_exclude_assistant=[], l_oneway_assistant = []):
        super().__init__(agents, messages, max_round)
        self.previous_speaker = None  # Keep track of the previous speaker
        self.graph = graph  # The graph depicting who are the next speakers available
        self.messages = messages  # The messages exchanged so far
        self.agents = agents  # The agents in the group chat
        self.memory = []  # The memory of the group chat, including retrieval information
        # self.chat_history = "" # The chat history of the group chat, it contains conversations only
        self.l_user = l_user # The list of users in the group chat
        self.l_exclude_assistant = l_exclude_assistant # The list of assistants that should be excluded from update memory and chat history
        self.l_oneway_assistant = l_oneway_assistant # The list of oneway assistants that should be excluded from update memory and chat history, such as message passing towards planner

    def update_memory(self, last_message):
        # Update the memory with the last message
        info=""
        if last_message["name"] in self.l_user:     # ["user_proxy", "chat_manager"]
            content=last_message["content"]
            # extract the user's query after <USER_QUERY> in the content string
            if '<USER_QUERY>' in content:
                user_query = content.split("<USER_QUERY>:\n")[1].split("<INFO>:\n")[0].strip()
                # extract the info after <INFO> in the content string
                info = content.split("<INFO>:\n")[1].split("<RESPONSE>:\n")[0].strip()
            else:
                user_query = content.strip()
            chat=f'User: {user_query}'
        elif last_message["name"] not in self.l_exclude_assistant:  # ["node_retrieve_agent"]
            content=last_message["content"]
            for oneway_assistant in self.l_oneway_assistant:    # ["planner_agent"]
                if oneway_assistant in content:
                    # condider to incorporate Node information into Chat_History
                    content=json.loads(content)
                    if "RESPONSE" in content:
                        self.memory.append({"chat": f'Node Retrieval: {content["RESPONSE"]}', "info": info})
                        return 
                    elif "NO_INFO_EXPLANATION" in content:
                        self.memory.append({"chat": f'Node Retrieval: {content["NO_INFO_EXPLANATION"]}', "info": info})
                        return
                    else:
                        return
            # extract the assistant's response after <RESPONSE> in the content string
            content=json.loads(content)
            response=content['RESPONSE']
            chat=f'TSG Copilot: {response}'
        else:
            return

        self.memory.append({"chat": chat, "info": info})

    def select_speaker(self, last_speaker, selector):       
        self.previous_speaker = last_speaker

        # Check if last message suggests a next speaker or termination
        last_message = self.messages[-1] if self.messages else None
        suggested_next = None
        
        if last_message:
            self.update_memory(last_message)
            if 'NEXT' in last_message['content']:
                # convert to json
                last_message = json.loads(last_message['content'])                
                suggested_next = last_message['NEXT']
                    
            elif 'TERMINATE' in last_message['content']:
                try:
                    return self.agent_by_name('User_proxy')
                except ValueError:
                    print(f"agent_by_name failed suggested_next: {suggested_next}")
                
        # Debugging print for the current previous speaker
        if self.previous_speaker is not None:
            print('Current previous speaker:', self.previous_speaker.name)

        # Selecting first round speaker
        if self.previous_speaker is None and self.graph is not None:
            eligible_speakers = [agent for agent in self.agents if self.graph.nodes[agent.name].get('first_round_speaker', False)]

        # Selecting successors of the previous speaker
        elif self.previous_speaker is not None and self.graph is not None:
            eligible_speaker_names = [target for target in self.graph.successors(self.previous_speaker.name)]
            eligible_speakers = [agent for agent in self.agents if agent.name in eligible_speaker_names]
            # print('Eligible speakers based on previous speaker:', eligible_speaker_names)

        else:
            eligible_speakers = self.agents

        # Debugging print for the next potential speakers
        # print(f"Eligible speakers based on graph and previous speaker {self.previous_speaker.name if self.previous_speaker else 'None'}: {[speaker.name for speaker in eligible_speakers]}")

        # Three attempts at getting the next_speaker
        # 1. Using suggested_next if suggested_next is in the eligible_speakers.name
        # 2. Using LLM to pick from eligible_speakers, given that there is some context in self.message
        # 3. Random (catch-all)
        next_speaker = None
        
        if eligible_speakers:
            # print("Selecting from eligible speakers:", [speaker.name for speaker in eligible_speakers])
            # 1. Using suggested_next if suggested_next is in the eligible_speakers.name
            if suggested_next in [speaker.name for speaker in eligible_speakers]:
                # print("suggested_next is in eligible_speakers")
                next_speaker = self.agent_by_name(suggested_next)
                
            else:
                msgs_len = len(self.messages)
                # print(f"msgs_len is now {msgs_len}")
                if len(self.messages) > 1:
                    # 2. Using LLM to pick from eligible_speakers, given that there is some context in self.message
                    # print(f"Using LLM to pick from eligible_speakers: {[speaker.name for speaker in eligible_speakers]}")
                    selector.update_system_message(self.select_speaker_msg(eligible_speakers))
                    _, name = selector.generate_oai_reply(self.messages + [{
                        "role": "system",
                        "content": f"Read the above conversation. Then select the next role from {[agent.name for agent in eligible_speakers]} to play. Only return the role.",
                    }])

                    # If exactly one agent is mentioned, use it. Otherwise, leave the OAI response unmodified
                    mentions = self._mentioned_agents(name, eligible_speakers)
                    if len(mentions) == 1:
                        name = next(iter(mentions))
                        next_speaker = self.agent_by_name(name)

                if next_speaker is None:
                    # 3. Random (catch-all)
                    next_speaker = random.choice(eligible_speakers)
                    
                
            print(f"Selected next speaker: {next_speaker.name}")

            return next_speaker
        else:
            # Cannot return next_speaker with no eligible speakers
            raise ValueError("No eligible speakers found based on the graph constraints.")