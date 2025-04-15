import conf
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest


def get_words(page, line):
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result


def _in_span(word, spans):
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False

def getDiClient() -> DocumentIntelligenceClient:
    endp = conf.AZURE_DOC_INTEL_ENDPOINT
    key = conf.AZURE_DOC_INTEL_API_KEY
    
    cli = DocumentIntelligenceClient(
        endpoint=endp,
        credential=AzureKeyCredential(key))

    return cli

def getFormattedData(path, result: AnalyzeResult):
    paginas = []
    for i, page in enumerate(result.pages):
        pagina = {
            "num": i,
            "text": "",
            "tables": ""
        }

        pageText = ""

        if page.lines:
            for line in page.lines:
                words = get_words(page, line)
                for word in words:
                    
                    if word.confidence <= conf.WORD_THRESHOLD_CONFIDENCE:
                        pageText = pageText + conf.INVALID_WORD_TOKEN + " "
                    else:
                        pageText = pageText + word.content + " "
                    #no opt por bug VS
        pagina["text"] = pageText

        pageTables = ""
        
        if result.tables:
            for table_idx, table in enumerate(result.tables):
                
                tableHeader = f"T{table_idx}-{table.row_count}-{table.column_count}$"
                pageTables = tableHeader

                for cell in table.cells:
                    pageTables = pageTables + f"{cell.row_index}-{cell.column_index}-{cell.content}$"
        
        pagina["tables"] = pageTables

        paginas.append(pagina)

    pdf = {
        "path": path,
        "pages": paginas
    }

    return pdf

def intelligentExtractPdf(pdfPath):

    with open(pdfPath, "rb") as pdf_file:
        pdf_bytes = pdf_file.read()

    cli = getDiClient()

    poller = cli.begin_analyze_document(
        "prebuilt-layout",
        AnalyzeDocumentRequest(bytes_source=pdf_bytes)
    )

    result: AnalyzeResult = poller.result()

    paginas = getFormattedData(pdfPath, result)

    return paginas


