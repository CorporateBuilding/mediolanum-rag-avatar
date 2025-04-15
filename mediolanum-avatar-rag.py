import streamlit as st
import time

import graph.automataSummarizer as aut
import agents.agents as ag

# Inicialización de la aplicación y configuración del estado de la sesión
def init_app():
    st.set_page_config(
        page_title="Asistente RAG",
        page_icon="🤖",
        layout="wide"
    )
    
    if 'initialized' not in st.session_state:
        st.session_state['initialized'] = False
        st.session_state['messages'] = []

    if not st.session_state['initialized']:
        try:
            st.session_state['initialized'] = True
            
        except Exception as e:
            st.error(f"Error inicializando servicios: {str(e)}")
            return False
    
    return True

# Muestra los mensajes del chat con márgenes
def display_chat_messages():
    col1, col2, col3 = st.columns([1, 8, 1])
    with col2:
        for message in st.session_state['messages']:
            if message["role"] == "user":
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])            
            else:
                with st.chat_message(message["role"]):
                    st.video(message["content"], format="video/mp4", autoplay=False, muted=False)


if __name__ == "__main__":  
    if not init_app():
        print("Error initializing agent. Exiting...")
        exit(1)

    # Se configura la interfaz principal
    st.title("🤖 Asistente RAG")

    display_chat_messages()  # Mostrar mensajes después de la entrada del usuario

    query = ""
    duration = ""
        
    c1, cont, c10 = st.columns([1,8,1])

    query = st.chat_input("Escribe tu consulta aquí...")
    limit, col2 = st.columns([1,5])
    with limit:
        duration = st.selectbox("Limit length (minutes)", ["No", "15'", "30'"])
        

    if query:
        
        FILEPATH = ""

        with cont:
            st.session_state['messages'].append({"role": "user", "content": query})
            
            with st.chat_message("user"):
                st.markdown(query)
            

            with st.status("Procesando datos...", expanded=True) as status:

                st.write(f"Recabando información en la base de datos...")

                text = "Error"

                try:
                    st.write(f"Generando curso...")

                    tiempo = 0
                    if duration == "30'":
                        tiempo = 30
                    elif duration == "15'":
                        tiempo = 15

                    text = aut.generateBigResponse(query, tiempo)

                    st.markdown(text)
                    text = text[:1750]
                    # # st.session_state['messages'].append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"Error al generar respuesta: {str(e)}"
                    st.error(error_msg)
                    st.session_state['messages'].append({"role": "assistant", "content": error_msg})

                if text == "Error":
                    status.update(label="Error generando curso", state="error", expanded=False)

                else:

                    st.write(f"Generando avatar")
                    
                    id_video = ag.getVideo(text)
                    print(id_video)


                    if id_video == -1:
                        st.write(f"Error generando vídeo")
                        status.update(label="Error con el curso!", state="error", expanded=False)

                    else:
                        
                        status_placeholder = st.empty()
                        path = ""
                        i=1
                        video_path = "video"

                        while True:
                            import os
                            os.makedirs(video_path, exist_ok=True)  # Crea el directorio si no existe
                            path = ag.checkVideo(video_path, id_video)

                            if path != -1 and path != -2:
                                break


                            status_placeholder.write(f"Llamando al agente" + "."*(i%4))

                            time.sleep(5)
                            i=i+1
                        
                        FILEPATH = path

                        status.update(label="¡Curso generado!", state="complete", expanded=False)
                        st.session_state['messages'].append({"role": "assistant", "content": FILEPATH})    

            with st.chat_message("assistant"):
                st.video(FILEPATH, format="video/mp4", autoplay=True, muted=False)
                        
