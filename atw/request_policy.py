"""One retry budget for every Agentic VLM request, regardless of client."""
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 20


class RequestRetries:
    """Retry unchanged evidence at most three times after the initial attempt."""
    preserve_messages = True

    def __init__(self, wait):
        self.wait = wait
        self.retries = 0
        self.context = {}

    def retry(self, kind, error):
        if self.retries >= MAX_RETRIES:
            return False
        self.retries += 1
        self.wait(RETRY_DELAY_SECONDS)
        return True
