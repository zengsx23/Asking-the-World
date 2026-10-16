"""Apply the Agentic policy to reused clients, including copied thread contexts."""
from functools import wraps
from time import sleep

from atw.request_policy import RequestRetries


def agentic_requests(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        from atw.shared.vlm_usage import REQUEST_RETRY_FACTORY
        token = REQUEST_RETRY_FACTORY.set(lambda: RequestRetries(sleep))
        try:
            return function(*args, **kwargs)
        finally:
            REQUEST_RETRY_FACTORY.reset(token)
    return wrapped
