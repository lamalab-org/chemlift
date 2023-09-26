from langchain.cache import SQLiteCache
import langchain
import pytest
import dotenv

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

dotenv.load_dotenv("../.env", override=True)


@pytest.fixture()
def get_llm():
    from langchain.llms import OpenAI

    llm = OpenAI(model_name="text-davinci-002")
    return llm
