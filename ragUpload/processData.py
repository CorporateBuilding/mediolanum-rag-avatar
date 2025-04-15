from PyPDF2 import PdfReader
import datetime
import streamlit as st
import tempfile
import os
import time
import conf
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from extractText.textProcessing import extractPdfsFromDir
from langchain.text_splitter import RecursiveCharacterTextSplitter
from extractText.textProcessing import extract_from_pdf
from extractText.advancedProcessing import intelligentExtractPdf

# Devuelve pdf con la siguiente estructura pdf = { "path": path, "pages": [ num, text, tables] }}
def extractSelectorPdf(pdfPath):
    pdfData = extract_from_pdf(pdfPath) #TODO DIFERENCIAR POR DOCINTELLIGENCE MEJOR
    detected_characters = sum(len(page["text"]) for page in pdfData["pages"])
    pages = len(pdfData)

    if detected_characters >= pages*conf.UMBRAL_MIN_CHARS_PER_PAGE:
        print("\t\tPROCESADO NORMAL")
    else:
        print("\t\tPROCESANDO DEEP")
        pdfData = intelligentExtractPdf(pdfPath) #TODO quitar
        exit("ERRORRORORORO")
    
    return pdfData

# Function that extracts metadata from PDF file path
# Parameters: pdf_path (str) - Path to the PDF file
# Returns: dict - Metadata including filename, path, size and dates
def extract_filename_metadata(pdf_path):
    full_filename = os.path.basename(pdf_path)
    filename_without_ext = os.path.splitext(full_filename)[0]
    file_extension = os.path.splitext(full_filename)[1]
    absolute_path = os.path.abspath(pdf_path)
    
    file_stats = os.stat(pdf_path)
    file_size = file_stats.st_size
    last_modified = datetime.datetime.fromtimestamp(file_stats.st_mtime).isoformat()
    
    metadata = {
        "filename": full_filename,
        "filename_without_extension": filename_without_ext,
        "file_extension": file_extension,
        "file_path": absolute_path,
        "file_size_bytes": file_size,
        "last_modified": last_modified,
        "extraction_time": datetime.datetime.now().isoformat()
    }
    
    return metadata

def getJustTextFromPdf(pdfStruct):
    string = ""
    
    for page in pdfStruct["pages"]: 
        string += page["text"] 
    
    return string

import re

def getTextFromVtt(vttPath):
    texto = ""
    
    with open(vttPath, "r", encoding="utf-8") as f:
        texto = f.read()

    regex = r"\n\n[0-9a-f]{8}-([0-9a-f]{4}-){3}[0-9a-f]{12}-\d*\n[0-9:.]* --> [0-9:.]*\n"
    result = re.sub(regex, "", texto)
    
    return result

import re
import uuid

# Function that loads text vectors into a QDrant database
# Parameters: text (str), embedder, qdrant, collection_name (str), pdf_metadata (dict, optional)
# Returns: int - Number of chunks added
def uploadQdrantFromText(text, embedder, qdrant, collection_name, pdf_metadata=None, page_id_re : str = conf.PAGE_IDENTIFIER_RE):
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=250)

    collectionExists = qdrant.collection_exists(collection_name=collection_name)
    
    if not collectionExists:
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )
    
    chunks = text_splitter.split_text(text)
    embeddings = embedder.embed_documents(chunks)

    total_points = qdrant.get_collection(collection_name).points_count
    
    points = []

    for id, (vector, chunk) in enumerate(zip(embeddings, chunks)):
        payload = {"text": chunk}

        num_pags = re.findall(page_id_re, chunk) #EXP REGULAR que muestra los números de página TODO: mejores formas de hacerlo

        if(num_pags != []):
            payload["pages"] = str(num_pags[0]) + "-" + str(num_pags[len(num_pags)-1])
        
        if pdf_metadata:
            payload["metadata"] = pdf_metadata
            payload["source_file"] = pdf_metadata["filename"]
        
        points.append({
            "id": id + total_points,
            "vector": vector,
            "payload": payload
        })
    qdrant.upsert(
        collection_name=collection_name,
        points=points
    )
    
    return len(chunks)

#Pre: dir = directorio del que extraer PDFs relative path from this function's file
#Post: List con SOLO TEXTO de PDFs TODO IMAGENES
def extractDataFromDir(dir: str, embedder, qdrant, collection_name):
    
    currentDir = os.getcwd()
    dataDir = os.path.join(currentDir, dir)
    dataList = []; pdfList = []
    totalFiles = len(os.listdir(dataDir))

    for i, file in enumerate(os.listdir(dataDir)):
        
        print(f"PROCESANDO {i+1}/{totalFiles}: {file}")
        filePath = os.path.join(dataDir, file)
        
        if not os.path.isfile(filePath):
            print("ERROR NOT FILE: ", filePath)
            continue

        metadata = extract_filename_metadata(filePath)
        text = ""

        if file.endswith(".pdf"):
            print("PDF")
            pdfStruct = extractSelectorPdf(filePath)
            text = getJustTextFromPdf(pdfStruct)
            
        elif file.endswith(".vtt"):
            print("SUBTITULOS")
            text = getTextFromVtt(filePath)

        uploadQdrantFromText(text, embedder, qdrant, collection_name, metadata, conf.PAGE_IDENTIFIER_RE)

    