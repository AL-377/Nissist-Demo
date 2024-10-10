class OutputResultsError(Exception):
    def __init__(self, message="Results should be outputted."):
        self.message = message
        super().__init__(self.message)

class DeleteConversationError(Exception):
    def __init__(self, message="Conversation is finished, should be deleted."):
        self.message = message
        super().__init__(self.message)

class MitigateConversationError(Exception):
    def __init__(self, message="Incident is mitigated, should stop."):
        self.message = message
        super().__init__(self.message)