import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from pinecone import Pinecone

# Cargar variables de entorno
load_dotenv()

# Obtener claves de entorno de forma segura
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
pinecone_namespace = os.getenv("PINECONE_NAMESPACE")


# Inicializar Pinecone
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(pinecone_index_name)

# Crear embeddings
embeddings = OpenAIEmbeddings(
    api_key=openai_api_key,
    model="text-embedding-3-small"
)

# Conectar con el vector store de Pinecone
vector_store = PineconeVectorStore(
    index=index,
    embedding=embeddings,
    text_key="text",
    namespace=pinecone_namespace
)

# Crear cadena de pregunta-respuesta
retriever = vector_store.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0),
    retriever=retriever
)

# Ejecutar consulta
query = "¿Qué puesto hay?"
response = qa_chain.invoke(query)
print(response)