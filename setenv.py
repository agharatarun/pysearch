import getpass
import os

os.environ["TOGETHER_API_KEY"] = getpass.getpass()

# (optional) LangSmith to inspect inside your chain or agent.
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
