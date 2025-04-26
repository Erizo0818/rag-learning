from langchain_community.document_loaders import PyPDFLoader

import asyncio

file_path = "../example_data/data_example.pdf"
async def load_pages():
    loader = PyPDFLoader(file_path)
    pages = []
    async for page in loader.alazy_load():
        pages.append(page)
    return pages

async def main():
    pages = await load_pages()
    print(pages)

loader = PyPDFLoader(file_path)
pages = loader.load()
# print(f"Total pages: {len(pages)}")
# print(f"{pages[0].metadata}\n")
# print(pages[0].page_content)
from google import genai

client = genai.Client(api_key="GEMINI_API_KEY")

result = client.models.embed_content(
        model="gemini-embedding-exp-03-07",
        contents="How does alphafold work?",
)

print(result.embeddings)


from langchain_core.vectorstores import InMemoryVectorStore

from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# print(embeddings.embed_query("Hello, world!"))
