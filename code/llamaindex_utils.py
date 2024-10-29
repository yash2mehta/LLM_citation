from llama_index.core import ServiceContext
from llama_index.core import VectorStoreIndex, SimpleKeywordTableIndex, RAKEKeywordTableIndex#, KeywordTableIndex
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import os

def set_service_context(key=None):
    if key is None:
        key = os.environ.get('OPENAI_API_KEY')
    service_context = ServiceContext.from_defaults(llm=OpenAI(api_key=key), embed_model=OpenAIEmbedding(api_key=key))
    return service_context

def get_index(index_name):
    Index = None
    if index_name == 'VectorStoreIndex':
        Index = VectorStoreIndex
    elif index_name == 'SimpleKeywordTableIndex':
        Index = SimpleKeywordTableIndex
    elif index_name == 'RAKEKeywordTableIndex':
        Index = RAKEKeywordTableIndex
    else:
        print('index name not imported')
    return Index
