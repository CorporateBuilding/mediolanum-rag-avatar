#!/usr/bin/env python
# coding: utf-8

# ### Import Libs

# In[224]:


from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client import QdrantClient
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
# from extractText.textProcessing import extractPdfsFromDir
from langchain.text_splitter import RecursiveCharacterTextSplitter

import conf
import os
import tiktoken
import re


# ### Create Needed Functions

# Initialize OpenAI embedder
embedder = AzureOpenAIEmbeddings(
    api_key= conf.AZURE_OPENAI_API_KEY,             
    azure_endpoint=conf.AZURE_OPENAI_ENDPOINT,  
    deployment=conf.AZURE_EMBEDDINGS_MODEL          
)

# Initialize OpenAI model endpoint
client = AzureOpenAI(    
    api_key=conf.AZURE_OPENAI_API_KEY,
    api_version=conf.AZURE_LLM_VERSION,
    azure_endpoint=conf.AZURE_LLM_ENDPOINT
)

# Initialize vector database
qdrant = QdrantClient(
    url = conf.QDRANT_API_URL, 
    api_key = conf.QDRANT_API_KEY,
)

# In[225]:


# Función para realizar la búsqueda de la query
def search_query(query, embedder, qdrant, k=1):
    # Generar el embedding de la consulta
    query_embedding = embedder.embed_query(query)
    
    # Buscar los k vectores más cercanos
    results = qdrant.query_points(
        collection_name=conf.QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=k
    )
    return results    


# In[226]:


def count_tokens(system_msg, model_name="gpt-3.5-turbo"):  
    # Clean the text by removing whitespace
    cleaned_text = re.sub(r'\s+', '', system_msg).strip()
    
    # Get the encoding for the specified
    encoding = tiktoken.encoding_for_model(model_name)

    # Count tokens
    original_tokens = encoding.encode(system_msg)
    original_count = len(original_tokens)
    print(f"El mensaje tiene {original_count} tokens.")
    cleaned_tokens = encoding.encode(cleaned_text)
    cleaned_count = len(cleaned_tokens)
    print(f"El mensaje LIMPIO tiene {cleaned_count} tokens.")
    
    return system_msg if original_count <= cleaned_count else cleaned_text


# In[227]:


def get_llm_response(client, system_msg, query, model="gpt-35-instant", block_size=70):
    # Message to answer
    
    messages = [
        {"role": "system", "content": system_msg},
        # {"role": "user", "content": query}
    ]
    
    # Call the API
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    
    return response.choices[0].message.content


# In[228]:


def generate_assistant_response(query, embedder, qdrant, client, invalid_word_token=conf.INVALID_WORD_TOKEN):
    # Get results from database
    results = search_query(query,embedder,qdrant)

    # Format all retrieved text chunks
    data_msg = "\n".join([str(p.payload["text"]) for p in results.points])
    question_msg = query
    
    # Create the system message with retrieved context
    system_msg = f"""
    Eres un asistente muy productivo especializado en banca cuyo objetivo es 
    proporcionar apoyo a las consultas de los trabajadores de un banco.
    Con la siguiente información, tienes que proporcionar una respuesta coherente 
    con un tono formal y serio. En caso de que la respuesta no se encuentre 
    en la siguiente información, no uses tu conocimiento general y responde 
    con "No tengo información relacionada". 
    Si lees {invalid_word_token} es porque no se ha podido reconocer la palabra 
    correctamente. Tu vida como asistente depende de estas respuestas

    Información: 
    {data_msg}
    Pregunta del usuario:
    {question_msg}
    """

    # Count tokens to process (optional)
    count_tokens(system_msg)
    
    # Generate and return response
    response = get_llm_response(client, system_msg, query)
    
    return response, results


# In[229]:


import json

def generate_questions_response(query, info, client):
    
    # Create the system message with retrieved context
    system_msg = f"""
    Eres un asistente muy productivo cuyo objetivo es generar un curso muy largo
    sobre una consulta que ha introducido el usuario. Se ha hecho ya una primera consulta al RAG que 
    contiene información y se han recabado los siguientes datos:
    
    Pregunta del usuario:
    {query}

    Información obtenida:
    {info}

    Tu objetivo es generar la mayor cantidad de preguntas que puedan ser hechas al RAG para 
    obtener la mayor cantidad de información posible y generar el curso. 
    El curso deberá tener una introducción, conceptos claves, partes, ejemplos, conclusión...
    pero sé muy creativo para generar el mayor número de preguntas. 
    Responde únicamente con el json de las preguntas. En él, habrá 3 grupos de preguntas, puesto
    que para optimizar se paralelizará la búsqueda en el RAG. No podrá haber preguntas repetidas
    en todo el json
    FORMATO: {{ "group1": ["q1", "q2"...], "group2": ["q1", "q2"...], "group3": ["q1", "q2"...]}}
    Emplea sólo la información proporcionada en este texto, no uses tu conocimiento general ni internet.

    """

    # Count tokens to process (optional)
    count_tokens(system_msg)
    
    # Generate and return response
    res = get_llm_response(client, system_msg, query)
    response = json.loads(res)
    
    print(response)
    
    return response["group1"], response["group2"], response["group3"]


# ### PDF Vectorization and query example

# In[231]:


from langgraph.types import Command, interrupt
from typing_extensions import TypedDict
from typing import List, Dict
import json

class State(TypedDict):
    query: str 
    adaptedQuery: str 
    limit: int
    
    retrievedFragments: list[str] 
    
    firstFragments: list[str]
    assistantResponse: str

    questions1: list[str]
    questions2: list[str]
    questions3: list[str]
    
    G1Hash: list[int]
    G2Hash: list[int]
    G3Hash: list[int]

    G1Texts: list[str]
    G2Texts: list[str]
    G3Texts: list[str]

    TotalHash: list[int]
    TotalTexts: list[str]

    courseQuery: str
    
    initIds: list[int]
    partsIds: Dict[str, List[int]]

    introduction: str
    development: str
    devTitPart: list[(str, str)]
    
    end: str

    fullSpeech: str

def ragAdapter(state: State):
    """Transforms the user's query into an adapted query for a RAG system"""

    print("RAG ADAPTER")

    system_msg = f"""
    Eres un experto desarrollador de RAGs (Retrieval-augmented generation) al que le han 
    encomendado la tarea de adaptar las preguntas realizadas por personas sin conocimientos
    tecnológicos al mejor modo que pueda ser entendido por un RAG. 
    
    -Por ejemplo, dado el input:

    ¡Hola!, muy buenos días, ¿cómo va todo? Siempre me han encantado el mundo de los ordenadores,
    tendría mucha curiosidad en conocer los datos sobre el mundo de la informática.

    -Debes responder únicamente con:
    "Información sobre el ámbito de la informática"
    
    PREGUNTA:
    {state['query']}
    """

    # Count tokens to process (optional)
    system_msg = count_tokens(system_msg)

    # print(system_msg)
    
    # # Generate and return response
    response = get_llm_response(client, system_msg, state["query"])
    
    return {"adaptedQuery": response}

def rag(state: State):
    """Searches into the Retrieval Augmented Generation system"""
    print("RAG")

    query = state["query"]

    assistantResponse, fragments = generate_assistant_response(query, embedder, qdrant, client, invalid_word_token=conf.INVALID_WORD_TOKEN)
    print(assistantResponse, fragments)

    return {"assistantResponse": assistantResponse, "firstFragments": fragments}


# In[232]:


def getQuestions(state: State):
    print("GET QUESTIONS")

    init, med, end = generate_questions_response(state["query"], state["firstFragments"], client)
    
    return {"questions1": init, "questions2": med, "questions3": end}


# In[233]:


#ante duda l2 más grande
def removeDuplicates(l1, l2, ll1, ll2):
    for e1 in l1:
        if e1 in l2:
            l1_index = l1.index(e1)
            l2_index = l2.index(e1)
            if len(l1) >= len(l2):
                l1.pop(l1_index)
                ll1.pop(l1_index)
            else:
                l2.pop(l2_index)
                ll2.pop(l2_index)
    
    return l1, l2, ll1, ll2


# In[234]:


def getG1(state: State):
    print("GET INTRO")

    hashes = []
    texts = []
    
    for question in state["questions1"]:
        result = search_query(question, embedder, qdrant, k=2)
        for point in result.points:
            id = point.id
            if id not in hashes:
                hashes.append(id)
                texts.append(point.payload["text"])

    return { "G1Texts": texts, "G1Hash": hashes}
    
def getG2(state: State):
    print("GET DEV")

    hashes = []
    texts = []
    
    for question in state["questions2"]:
        result = search_query(question, embedder, qdrant, k=2)
        for point in result.points:
            id = point.id
            if id not in hashes:
                hashes.append(id)
                texts.append(point.payload["text"])

    return { "G2Texts": texts, "G2Hash": hashes}

def getG3(state: State):
    print("GET CONC")


    hashes = []
    texts = []
    
    for question in state["questions3"]:
        result = search_query(question, embedder, qdrant, k=2)
        for point in result.points:
            id = point.id
            if id not in hashes:
                hashes.append(id)
                texts.append(point.payload["text"])

    return { "G3Texts": texts, "G3Hash": hashes}


def balancer(state: State):
    print("BALANCE")


    l1 = state["G1Hash"]; ll1 = state["G1Texts"]
    l2 = state["G2Hash"]; ll2 = state["G2Texts"]
    l3 = state["G3Hash"]; ll3 = state["G3Texts"]
    
    l1, l2, ll1, ll2 = removeDuplicates(l1, l2, ll1, ll2)
    l2, l3, ll2, ll3 = removeDuplicates(l2, l3, ll2, ll3)
    l1, l3, ll1, ll3 = removeDuplicates(l1, l3, ll1, ll3)
    
    return {
        "G1Hash": l1, "G1Texts": ll1,
        "G2Hash": l2, "G2Texts": ll2,
        "G3Hash": l3, "G3Texts": ll3
    }


# In[235]:


import json

def generate_course_json(query, info, client):
    
    # Create the system message with retrieved context
    LLM_MAX_INPUT = 16000
    SYS_LEN = 2000
    MAX_CHARS = LLM_MAX_INPUT - SYS_LEN
    data = info[:MAX_CHARS]

    system_msg = f"""
    Eres un asistente muy productivo cuyo objetivo es generar la estructura de un curso para un
    banco muy importante. El curso deberá tener mínimo una introducción, en la que nombrar lo que 
    se va a exponer brevemente; un desarrollo, compuesto por las distintas secciones que se 
    obtienen y deducen de la siguiente información; y una conclusión, que sirva como cierre.

    La información para generar el curso se te pasa con formato IDENTIFICADOR-CONTENIDO, de forma
    que para la estructura del curso, deberás indicar únicamente los id que formarán cada parte
    del curso, no el contenido asociado a ellos

    Tu salida será un json con el siguiente formato
    {{ 
        "intro": [idx, idy, idz...], 
        "development": [
            "tituloparte1" : [idf, idg, idh...],
            "tituloparte2" : [idr, ids, idt...],
            ...
        ]
        "conclusion": "conclusion text"
    }}

    Intenta ser descriptivo con los títulos de las partes ya que la información suministrada viene 
    de un RAG, por lo que seguramente en un mismo ID puede haber contenido de varias partes, y 
    como cada parte va a ser generada por un agente generador de textos independiente, el hecho de
    que compartan el mismo fragmento de información podría hacer que, con títulos poco descriptivos,
    se generasen partes de texto iguales en los agentes.
    La conclusión es una tarea complicada que no se puede hacer sin tener todo el contexto, realiza
    una larga conclusión, de unas 250 palabras que sirva como cierre del curso y que sirva también
    como un gran resumen de las ideas que se han podido plantear en el curso.

    No emplees información tuya propia, usa únicamente la proporcionada a continuación    

    Para generar el curso, la entidad bancaria ha preguntado:
    
    {query}

    Y se ha obtenido la siguiente información para generar el curso:

    {data}
    """

    # Count tokens to process (optional)
    count_tokens(system_msg)
    
    # Generate and return response
    res = get_llm_response(client, system_msg, query)
    print(res)
    response = json.loads(res)
    print(response)
    
    
    return response["intro"], response["development"], response["conclusion"], system_msg

def generate_course_structure(state: State):

    state["TotalHash"] = state["G1Hash"] + state["G2Hash"] + state["G3Hash"]
    state["TotalTexts"] = state["G1Texts"] + state["G2Texts"] + state["G3Texts"]

    totalList = zip(state["TotalHash"], state["TotalTexts"])

    courseFragmentsWithId= "\n\n".join(["ID:"+str(e1)+"\nCONTENT:"+e2 for e1, e2 in totalList])

    intro, parts, conclusion, state["courseQuery"] = generate_course_json(state["query"], courseFragmentsWithId, client)

    return {"initIds" : intro, "partsIds": parts, "end": conclusion}




# In[236]:


def generateIntro(state: State):
    print("GEN INTRO")

    query = state["query"]

    idList = state["initIds"]

    hashList = state["G1Hash"]+state["G2Hash"]+state["G3Hash"]
    textList = state["G1Texts"]+state["G2Texts"]+state["G3Texts"]

    # info = "\n\n".join([textList[hashList.index(elem)] for elem in idList])
    info = "\n\n".join([textList[hashList.index(elem)] for elem in idList if elem in hashList])


    LLM_MAX_INPUT = 16000
    SYS_LEN = 1100
    MAX_CHARS = LLM_MAX_INPUT - SYS_LEN
    data = info[:MAX_CHARS]

    system_msg = f"""
    Se está generando un curso para los empleados de una compañia muy importante
    y se te ha encomendado la tarea de realizar la introducción. Tienes un gran 
    conocimiento sobre la lengua y emplearás todos los recursos lingüisticos posibles
    para hacer una introducción larga y con un lenguaje muy experto. La introducción es
    una parte muy importante, pues anticipa los temas de los que se hablará en el desarrollo
    del curso. No aprofundices en exceso, pues posteriores secciones se encargarán de ello. 
    No utilices información externa, para la introducción emplea únicamente los fragmentos
    que se incluyen a continuación.
     
    FRAGMENTOS para realizar la introducción, que sea larga, por favor.

    {data}    

    Como dato informativo, la empresa ha realizado la siguiente pregunta como base
    para generar el curso, :

    {query}

    Tu vida como asistente depende de este texto.
    
    Responde únicamente con la introducción."""

    # Count tokens to process (optional)
    count_tokens(system_msg)
    
    # Generate and return response
    response = get_llm_response(client, system_msg, query)

    return {"introduction": response}


# In[237]:


def generatePart(titulo: str, idList: list[int], state: State):
    print("GEN PART", titulo)

    # idList = state["initIds"]

    hashList = state["G1Hash"]+state["G2Hash"]+state["G3Hash"]
    textList = state["G1Texts"]+state["G2Texts"]+state["G3Texts"]

    info = "\n\n".join([textList[hashList.index(elem)] for elem in idList if elem in hashList])

    LLM_MAX_INPUT = 16000
    SYS_LEN = 1700
    MAX_CHARS = LLM_MAX_INPUT - SYS_LEN
    data = info[:MAX_CHARS]

    system_msg = f"""
    Una compañía muy importante está realizando cursos para sus empleados. Cada uno 
    de ellos está formado por distintas partes independientes. En tu caso, como eres
    un gran escritor con un gran conocimiento sobre recursos lingüisticos, se te ha 
    encomendado la tarea de realizar una de estas partes.
 
    La parte que se te ha encargado es muy importante y tiene como título "{titulo}", y 
    para completarla tendrás que hacer uso de la siguiente información que se ha recabado:

    --INICIO INFORMACION--
    
    {data}

    --FIN INFORMACION--

    Tu respuesta es parte del desarrollo del curso, por lo que deberás aprofundizar todo
    lo posible en aquello que tenga que ver con "{titulo}". Ten en cuenta que la información
    puede incluir fragmentos que le serán enviados a otros agentes generadores de otras
    partes, por lo que cíñete únicamente al ámbito del título de tu parte, pues si no
    habrá información redundante y repetida en las partes del curso. 
    
    La respuesta deberá contener una introducción al tema a tratar con el título (en caso de haber
    mucha información además un resumen) y luego el desglose de los distintos apartados y
    secciones que tengan que ver con "{titulo}" con la información que se te ha adjuntado.
    Hazla lo más larga posible, siempre ciñéndote sólo al tema sobre el cual se te pregunta.

    Tu vida como asistente depende de este texto.
    
    Responde únicamente con la parte que se te ha preguntado."""

    # Count tokens to process (optional)
    count_tokens(system_msg)
    
    # Generate and return response
    response = get_llm_response(client, system_msg, state["query"])

    return response

def generateAllParts(state: State):
    totalDev = ""
    parts = []
    dictWithParts = state["partsIds"]
    for title, ids in dictWithParts.items():
        part = generatePart(title, ids, state)
        parts.append((title, part))
        totalDev = totalDev + "\n\n-" + title + "\n\n" + part + "\n"
            
    return {"development": totalDev, "devTitPart": parts}


# In[238]:
def reducePart(titulo: str, text: str, reduce: float):
    print("REDUCE PART")

    red = round(reduce, 1)

    LLM_MAX_INPUT = 16000
    SYS_LEN = 1100
    MAX_CHARS = LLM_MAX_INPUT - SYS_LEN
    data = text[:MAX_CHARS]

    system_msg = f"""
    Eres un agente superespecializado en la reducción de la longitud de un texto. Una 
    empresa muy importante ha encargado un curso con una parte que ha quedado demasiado
    larga y la debes reducir. El trabajo que realizas es muy importante, pues se harán 
    cursos formativos con la información que generes para millones de personas, por lo 
    que es necesario una reducción de calidad.

    En este caso, debes reducir la longitud del texto adjuntado exactamente un {red} por ciento.
    Intenta mantener la estructura del texto adjuntado lo más posible, pese a la reducción, y decir
    la mayor cantidad de información posible del texto original, dentro de lo posible.

    Como información, el título del texto que tienes que reducir es "{titulo}"

    --INITIO TEXTO A REDUCIR--

    {data}

    --FIN TEXTO A REDUCIR--

    Tu vida como asistente depende de este texto.
    
    Tu respuesta deberá contener únicamente el texto reducido, no respondas con nada más"""

    # Count tokens to process (optional)
    count_tokens(system_msg)
    
    # Generate and return response
    response = get_llm_response(client, system_msg, "")

    return response

def wait(state: State):
    totalSpeech = state['introduction'] + "\n" + state["development"] + "\n" + state["end"]

    return {"fullSpeech": totalSpeech}

def checkLenght(state: State):
    # totalChars = len(state["introduction"]) + len(state["development"]) + len(state["end"])
    totalChars = len(state["fullSpeech"])
    CHARS_LECTURA_NORMAL = 540 #chars/min
    speechMins = totalChars * 1.0 / CHARS_LECTURA_NORMAL
    MINS_UMBRAL = 3.5 #tiempo de introducción y conclusión

    limit = state["limit"]

    if(limit == 0):
        return {"limit": limit}
    else:
        #speechMins/(state["limit"] + MINS_UMBRAL) ES cuantos limit es el curso generado
        #^-1 es 
        reduceNum = 100.0 - (limit*1.0 + MINS_UMBRAL)/speechMins * 100 #cuanto hay q reducir %

        if reduceNum >= 12.0:

            totalSpeech = state["introduction"] +"\n\n"
            titParts = state["devTitPart"]

            for tit, text in titParts:
                reducedText = reducePart(tit, text, reduceNum)
                totalSpeech = totalSpeech + tit + "\n" + reducedText + "\n\n"

            totalSpeech = totalSpeech + state["end"]
            
            return {"fullSpeech" : totalSpeech}
        else:
            return {"limit": limit}


# In[239]:


from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display


# Build workflow
optimizer_builder = StateGraph(State)

#Add the nodes
# optimizer_builder.add_node("ragAdapter", ragAdapter)
optimizer_builder.add_node("rag", rag)
# optimizer_builder.add_node("validateRagsOutput", validateRagsOutput)
optimizer_builder.add_node("getQuestions", getQuestions)
# optimizer_builder.add_node("retrieveQuestions", retrieveQuestions)
optimizer_builder.add_node("getG1", getG1)
optimizer_builder.add_node("getG2", getG2)
optimizer_builder.add_node("getG3", getG3)

optimizer_builder.add_node("balancer", balancer)
optimizer_builder.add_node("generate_course_structure", generate_course_structure)

optimizer_builder.add_node("generateIntro", generateIntro)
optimizer_builder.add_node("generateAllParts", generateAllParts)

optimizer_builder.add_node("wait", wait)
optimizer_builder.add_node("checkLenght", checkLenght)


# optimizer_builder.add_edge(START, "ragAdapter")
optimizer_builder.add_edge(START, "rag")
# optimizer_builder.add_edge("ragAdapter", "rag")
optimizer_builder.add_edge("rag", "getQuestions")

optimizer_builder.add_edge("getQuestions", "getG1")
optimizer_builder.add_edge("getQuestions", "getG2")
optimizer_builder.add_edge("getQuestions", "getG3")

optimizer_builder.add_edge("getG1", "balancer")
optimizer_builder.add_edge("getG2", "balancer")
optimizer_builder.add_edge("getG3", "balancer")

optimizer_builder.add_edge("balancer", "generate_course_structure")

optimizer_builder.add_edge("generate_course_structure", "generateIntro")
optimizer_builder.add_edge("generate_course_structure", "generateAllParts")

optimizer_builder.add_edge("generateIntro", "wait")
optimizer_builder.add_edge("generateAllParts", "wait")
optimizer_builder.add_edge("wait", "checkLenght")


optimizer_builder.add_edge("checkLenght", END)


# Compile the workflow
optimizer_workflow = optimizer_builder.compile()


# # In[240]:


# state = optimizer_workflow.invoke({"query": "Querría que me explicases el modelo My World"})

def generateBigResponse(query : str, limit : int):

    state = optimizer_workflow.invoke({"query": query, "limit": limit})

    return state["fullSpeech"]
    # return state["introduction"]
