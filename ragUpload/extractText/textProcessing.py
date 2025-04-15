import os
from PyPDF2 import PdfReader
import conf
from extractText.advancedProcessing import intelligentExtractPdf

#Pre: pdf_path
#Post: List con {path {pag, texto, imagenes}} del PDF
def extract_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    
    paginas = []
    
    for i, page in enumerate(reader.pages):
        pagina = {
            "num": i,
            "text": page.extract_text(),
            "imgs": [],
        }
        count = 0
        try:
            if page.images:    
                for img in page.images:
                    pagina["imgs"].append(img.data)
                    count += 1
        except ValueError:
            pass  # No hay imágenes, continuar normalmente
        
        paginas.append(pagina)
    
    pdf = {
        "path": pdf_path,
        "pages": paginas
    }

    return pdf

#Pre: dir = directorio del que extraer PDFs
#Post: List con SOLO TEXTO de PDFs TODO IMAGENES
def extractPdfsFromDir(dir: str):
    
    pdfDatas = []
    totalPdfs = len(os.listdir(dir))

    for i, file in enumerate(os.listdir(dir)):
        pdfPath = os.path.join(dir, file)
        if file.endswith(".pdf") and os.path.isfile(pdfPath):
            pdfData = extract_from_pdf(pdfPath) #TODO DIFERENCIAR POR DOCINTELLIGENCE MEJOR
            detected_characters = sum(len(page["text"]) for page in pdfData)
            pages = len(pdfData)

            if detected_characters >= pages*conf.UMBRAL_MIN_CHARS_PER_PAGE:
                print("PROCESANDO PDF ", i+1, "/", totalPdfs, file)
                pdfDatas.append(pdfData)
            else:
                print("DEEP PROCESANDO PDF ", i+1, "/", totalPdfs, file)
                #pdfDatas.append(intelligentExtractPdf(pdfPath))
    
    return pdfDatas

#Pre: dir = directorio del que extraer PDFs
#Post: List con SOLO TEXTO de PDFs TODO IMAGENES
def extractVttsFromDir(dir: str):
    
    vttDatas = []
    totalPdfs = len(os.listdir(dir))

    for i, file in enumerate(os.listdir(dir)):
        pdfPath = os.path.join(dir, file)
        if file.endswith(".vtt") and os.path.isfile(pdfPath):
            pdfData = extract_from_pdf(pdfPath) #TODO DIFERENCIAR POR DOCINTELLIGENCE MEJOR
            detected_characters = sum(len(page["text"]) for page in pdfData)
            pages = len(pdfData)

            if detected_characters >= pages*conf.UMBRAL_MIN_CHARS_PER_PAGE:
                print("PROCESANDO PDF ", i+1, "/", totalPdfs, file)
                vttDatas.append(pdfData)
            else:
                print("DEEP PROCESANDO PDF ", i+1, "/", totalPdfs, file)
                vttDatas.append(intelligentExtractPdf(pdfPath))
    
    return vttDatas

