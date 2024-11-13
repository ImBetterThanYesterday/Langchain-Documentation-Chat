import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from langchain.chains import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chains import create_retrieval_chain
import os

# Configuración de Streamlit
st.set_page_config(page_title="LangChain Documentation Assistant", layout="wide")

# Inicializar el historial de chat en la sesión
if "history" not in st.session_state:
    st.session_state.history = []

# Directorio para persistencia
persist_directory = "db"
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
# URLs de documentación de LangChain
urls_list = [
    "https://python.langchain.com/docs/get_started/introduction",
    "https://python.langchain.com/docs/modules/model_io/",
    "https://python.langchain.com/docs/modules/memory/",
    "https://python.langchain.com/docs/modules/data_connection/",
]

# Cargar y procesar documentos
@st.cache_resource
def load_and_process_documents():
    loader = WebBaseLoader(urls_list)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    documents = text_splitter.split_documents(docs)
    
    # Crear y persistir vectorstore
    vectorstore = Chroma.from_documents(
        documents=documents,
        collection_name="langchain-docs",
        embedding=OllamaEmbeddings(model='nomic-embed-text'),
        persist_directory=persist_directory,
    )
    vectorstore.persist()
    
    return vectorstore

# Cargar vectorstore
vectorstore = load_and_process_documents()

# Configurar el modelo y el prompt
llm = ChatOllama(model="granite3-moe:1b")

# System prompt
SYSTEM_PROMPT = """You are a helpful AI assistant specialized in LangChain documentation. 
Use the provided context to answer questions about LangChain. 
If you're not sure about something, say so. Always base your answers on the provided documentation context.
Keep your answers focused on LangChain-specific information."""

# Primero, creamos el prompt para el retriever
retriever_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that helps users find relevant information.
    Consider the chat history when finding relevant information to maintain context."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
])

# Luego, creamos el prompt para el documento
document_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", """Remember to consider the previous conversation when providing answers.
    If the user refers to something mentioned earlier, use that context."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("system", "Context: {context}")
])

# Crear el chain para documentos
document_chain = create_stuff_documents_chain(
    llm,
    document_prompt
)

# Crear el retriever consciente del historial
retriever = vectorstore.as_retriever()
retriever_chain = create_history_aware_retriever(
    llm,
    retriever,
    retriever_prompt
)

# Crear el chain de recuperación
chain = create_retrieval_chain(
    retriever_chain,
    document_chain
)

# Agregar un contador de mensajes en la sesión
if "message_count" not in st.session_state:
    st.session_state.message_count = 0

# Interfaz de Streamlit
st.title("💬 LangChain Documentation Assistant")

# Sidebar
st.sidebar.title("LangChain Documentation Assistant")
st.sidebar.info("""
Este chatbot está especializado en responder preguntas sobre LangChain 
usando la documentación oficial como fuente de información.
""")

# Agregar métricas de la conversación
st.sidebar.markdown("### Estadísticas de la Conversación")
st.sidebar.metric("Mensajes en la conversación actual", len(st.session_state.history) // 2)
st.sidebar.metric("Mensajes totales", st.session_state.message_count)

# Agregar información sobre la base de datos vectorial
st.sidebar.markdown("### Estado de la Base de Datos")
st.sidebar.info(f"""
📁 Directorio de persistencia: {persist_directory}
💾 Base de datos vectorial: Chroma
🔤 Modelo de embeddings: nomic-embed-text
""")

# Mostrar el historial de manera más detallada
st.markdown("### Historial de la Conversación")
for i, message in enumerate(st.session_state.history):
    if isinstance(message, HumanMessage):
        with st.chat_message('user'):
            st.write(message.content)
    else:
        with st.chat_message('assistant'):
            st.write(message.content)

# Botón para limpiar conversación
if st.button("Limpiar Conversación"):
    st.session_state.history = []
    st.rerun()

# Agregar botón para exportar historial
if st.sidebar.button("Exportar Historial"):
    history_text = "\n\n".join([
        f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
        for m in st.session_state.history
    ])
    st.sidebar.download_button(
        label="Descargar Historial",
        data=history_text,
        file_name="historial_chat.txt",
        mime="text/plain"
    )

# Input del usuario
if user_input := st.chat_input("Hazme una pregunta sobre LangChain"):
    # Incrementar el contador de mensajes
    st.session_state.message_count += 1
    
    # Mostrar mensaje del usuario
    with st.chat_message('user'):
        st.write(user_input)

    # Preparar el historial para el modelo
    chat_history = []
    for message in st.session_state.history:
        if isinstance(message, HumanMessage):
            chat_history.append(("human", message.content))
        else:
            chat_history.append(("assistant", message.content))

    # Mostrar el historial que se está usando (para debug)
    st.sidebar.markdown("### Historial usado en la consulta actual")
    st.sidebar.info(f"Número de mensajes en el historial: {len(chat_history)}")
    if st.sidebar.checkbox("Mostrar historial detallado"):
        for role, content in chat_history:
            st.sidebar.text(f"{role}: {content[:100]}...")

    # Obtener respuesta
    response = chain.invoke({
        "chat_history": chat_history,
        "input": user_input
    })

    # Mostrar respuesta del asistente
    with st.chat_message('assistant'):
        st.write(response['answer'])

    # Actualizar historial
    st.session_state.history.append(HumanMessage(content=user_input))
    st.session_state.history.append(AIMessage(content=response['answer']))