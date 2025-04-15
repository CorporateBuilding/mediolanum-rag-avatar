import streamlit as st

HEYGEN_API_KEY=st.secrets["st_HEYGEN_API_KEY"]

##### Qdrant

QDRANT_API_URL = st.secrets["st_QDRANT_API_URL"]

QDRANT_API_KEY = st.secrets["st_QDRANT_API_KEY"]

QDRANT_COLLECTION_NAME=st.secrets["st_QDRANT_COLLECTION_NAME"]

##### Document Intelligence

AZURE_DOC_INTEL_ENDPOINT=st.secrets["st_AZURE_DOC_INTEL_ENDPOINT"]

AZURE_DOC_INTEL_API_KEY=st.secrets["st_AZURE_DOC_INTEL_API_KEY"]

##### OPENAI

AZURE_OPENAI_ENDPOINT=st.secrets["st_AZURE_OPENAI_ENDPOINT"]
 #solo url principal Ej: https://xxx-xxx-xxx.openai.azure.com/
AZURE_OPENAI_API_KEY=st.secrets["st_AZURE_OPENAI_API_KEY"]

AZURE_EMBEDDINGS_MODEL=st.secrets["st_AZURE_EMBEDDINGS_MODEL"]
 #tested with ada-002

AZURE_LLM_MODEL=st.secrets["st_AZURE_LLM_MODEL"]
 #tested with gpt-35-turbo
AZURE_LLM_ENDPOINT=st.secrets["st_AZURE_LLM_ENDPOINT"]
 #url completa Ej: https://xxx-xxx-xxx.openai.azure.com/openai/deployments/.../.../...api-version=xxxx-xx-xx
AZURE_LLM_VERSION=st.secrets["st_AZURE_LLM_VERSION"]
 

#####Umbral y texts

UMBRAL_MIN_CHARS_PER_PAGE=20
WORD_THRESHOLD_CONFIDENCE=0.6
INVALID_WORD_TOKEN="<WORD>"
TABLE_STRUCTURE="{T{num_tabla}-{num_filas}-{num_cols}${{fila_celda}-{fila_columna}-{content}$}*}*"
PAGE_IDENTIFIER="<(P{})>"
PAGE_IDENTIFIER_RE=r"<\(P(\d+)\)>"
