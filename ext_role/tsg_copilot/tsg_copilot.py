import json

import requests
from injector import inject

from taskweaver.logging import TelemetryLogger
from taskweaver.memory import Memory, Post
from taskweaver.module.event_emitter import SessionEventEmitter
from taskweaver.module.tracing import Tracing
from taskweaver.role import Role
from taskweaver.role.role import RoleConfig, RoleEntry


class TSGCopilotConfig(RoleConfig):
    def _configure(self):
        self._set_name("tsg_copilot")
        self.tsg_copilot_url = self._get_str("tsg_copilot_url", "http://127.0.0.1:2000/api/tsg_copilot")


class TSGCopilot(Role):
    @inject
    def __init__(
        self,
        config: TSGCopilotConfig,
        logger: TelemetryLogger,
        tracer: Tracing,
        event_emitter: SessionEventEmitter,
        role_entry: RoleEntry,
    ):
        super().__init__(config, logger, tracer, event_emitter, role_entry)

    def reply(self, memory: Memory, **kwargs) -> Post:
        rounds = memory.get_role_rounds(
            role=self.alias,
            include_failure_rounds=False,
        )

        # obtain the query from the last round
        last_post = rounds[-1].post_list[-1]

        # call tsg_copilot
        tsg_copilot_response = self.tsg_copilot_API(last_post.message, memory.session_id)

        post_proxy = self.event_emitter.create_post_proxy(self.alias)
        post_proxy.update_send_to(last_post.send_from)
        post_proxy.update_message(tsg_copilot_response)

        return post_proxy.end()

    def tsg_copilot_API(self, message: str, conversation_id: str) -> str:
        # Define the API endpoint and parameters
        input_json = {
            "conversation_id": conversation_id,
            "query": message,
        }

        response = requests.post(self.config.tsg_copilot_url, json=input_json)

        if response.status_code != 200:
            raise Exception(f"Failed to call TSG Copilot API: {response.status_code}")
        else:
            content = response.content.decode("utf-8")
            content = json.loads(content)["response"]
            return content
