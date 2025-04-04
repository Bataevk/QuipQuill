import sys
import os


from dotenv import load_dotenv
import os


from langchain.chat_models import init_chat_model

load_dotenv()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from minirag import MiniRAG, QueryParam
from minirag.llm.hf import (
    hf_model_complete,
    hf_embed,
)

# from lightrag import LightRAG
from minirag import MiniRAG
from minirag.utils import EmbeddingFunc
from transformers import AutoModel, AutoTokenizer

from lightrag.llm.openai import openai_complete_if_cache

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


WORKING_DIR = os.getenv("WORKING_DIR")
DATA_PATH = os.getenv("DATA_PATH")
QUERY_PATH = os.getenv("QUERY_PATH")
OUTPUT_PATH = os.getenv("OUTPUT_PATH")
LLM_MODEL = os.getenv("RAG_LLM_MODEL")
BASE_URL = os.getenv("RAG_BASE_URL")


print("USING LLM:", LLM_MODEL)
print("USING WORKING DIR:", WORKING_DIR)
print("USING DATA PATH:", DATA_PATH)
print("USING QUERY PATH:", QUERY_PATH)


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)



async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    return await openai_complete_if_cache(
        LLM_MODEL,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=None,    # os.getenv("OPENAI_API_KEY"),
        base_url=BASE_URL,
        **kwargs
    )

def init_rag():
    return MiniRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        # llm_model_func= LLM_MODEL,
        llm_model_max_token_size=4096,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=512,
            func=lambda texts: hf_embed(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL),
                embed_model=AutoModel.from_pretrained(EMBEDDING_MODEL),
            ),
        ),
        chunk_token_size=256,
    )



# Now indexing
# def find_txt_files(root_path):
#     txt_files = []
#     for root, dirs, files in os.walk(root_path):
#         for file in files:
#             if file.endswith(".txt"):
#                 txt_files.append(os.path.join(root, file))
#     return txt_files

def find_txt_files(files_dir):
    txt_files = []
    for file in os.listdir(files_dir):
        if file.endswith(".txt"):
            txt_files.append(os.path.join(files_dir, file))
    return txt_files


if __name__ == "__main__":
    rag = init_rag()
    WEEK_LIST = find_txt_files(DATA_PATH)

    print("FOUND WEEKS:", WEEK_LIST)

    for WEEK in WEEK_LIST:
        id = WEEK_LIST.index(WEEK)
        print(f"{id}/{len(WEEK_LIST)}")
        with open(WEEK, encoding='utf-8') as f:
            rag.insert(f.read())