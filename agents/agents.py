import conf as c
import json
import requests
from datetime import datetime
import os



def getVideo(text: str):
    """Devuelve ID vídeo o -1 si error. Con ID se puede comprobar si ha sido ya preparado con función check"""
    # URL de la API
    url = "https://api.heygen.com/v2/video/generate"

    # Cabeceras de la solicitud
    headers = {
        'X-Api-Key': c.HEYGEN_API_KEY,  # Reemplaza con tu clave de API
        'Content-Type': 'application/json'
    }

    # Cuerpo de la solicitud (datos JSON)
    data = {
        "video_inputs": [
            {
                "character": {
                    "type": "avatar",
                    "avatar_id": "Annie_expressive_public",
                    "avatar_style": "normal"
                },
                "voice": {
                    "type": "text",
                    "input_text": text,
                    "voice_id": "3fac0e13ef4d42c0a30bc20e524ae43d",
                    "speed": 1.1
                }
            }
        ],
        "dimension": {
            "width": 1280,
            "height": 720
        }
    }

    # Realizar la solicitud POST
    response = requests.post(url, headers=headers, json=data)

    # Verificar la respuesta
    if response.status_code == 200:
        print("Video generado correctamente")
        # Puedes guardar la respuesta si es necesario, por ejemplo:
        # with open("output_video.mp4", "wb") as f:
        #     f.write(response.content)
        rcv = response.json()
        id_video = rcv["data"]["video_id"]
        return id_video
    else:
        print(f"Error al generar el video: {response.status_code}, {response.text}")
        return -1
    
def checkVideo(relPath: str, id: str):
    url = f"https://api.heygen.com/v1/video_status.get?video_id={id}"

    headers = {
        "accept": "application/json",
        "x-api-key": c.HEYGEN_API_KEY
    }

    response = requests.get(url, headers=headers).json()
    status = response["data"]["status"]
    print(response["data"])

    if status == "completed":
        print("OKAY")
        videoUrl = response["data"]["video_url"]
        response = requests.get(videoUrl, stream=True)

        # Verificar si la solicitud fue exitosa
        if response.status_code == 200:
            # Abrir un archivo para guardar el video
            filename = str(datetime.now().strftime("%Y%m%d%H%M%S")) + ".mp4"
            filePath = os.path.join(relPath, filename)
            with open(filePath, 'wb') as f:
                # Escribir el contenido del video en el archivo
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)

            print("¡Video descargado correctamente!")
            return filePath
        else:
            print(f"Error al descargar el video: {response.status_code}")
            return -2

    else:
        print(status)
        return -1
    