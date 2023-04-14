import openai
import os

our_openai = openai
our_openai.organization = os.environ.get("OPENAI_ORG", None)
our_openai.api_key = os.environ.get("OPENAI_API_KEY", None)

__all__ = ["our_openai"]