import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model
import dotenv

dotenv.load_dotenv()
try:
    # load environment variables from .env file (requires `python-dotenv`)
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

os.environ["LANGSMITH_TRACING"] = "true"
if "LANGSMITH_API_KEY" not in os.environ:
    os.environ["LANGSMITH_API_KEY"] = getpass.getpass(
        prompt="Enter your LangSmith API key (optional): "
    )
if "LANGSMITH_PROJECT" not in os.environ:
    os.environ["LANGSMITH_PROJECT"] = getpass.getpass(
        prompt='Enter your LangSmith Project Name (default = "default"): '
    )
    if not os.environ.get("LANGSMITH_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = "default"
# if "OPENAI_API_KEY" not in os.environ:
#     os.environ["OPENAI_API_KEY"] = getpass.getpass(
#         prompt="Enter your OpenAI API key (required if using OpenAI): "
#     )


if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


model = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

messages = [
    SystemMessage("Translate the following from English into Italian"),
    HumanMessage("hi!"),
]

# print(model.invoke(messages).content)

from langchain_core.prompts import ChatPromptTemplate

system_template = "Translate the following from {language} into {target_language}"
prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_template),
    ("user", "{text}"),
])

prompt = prompt_template.format(language="English", target_language="Chinese", text="I love programming in Python")

print(model.invoke(prompt).content)