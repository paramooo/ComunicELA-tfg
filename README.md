# ComunicELA: Software de Asistencia para la Comunicación en Pacientes con Esclerosis Lateral Amiotrófica

## Resúmen

## Abstract

## ¿Qué es necesario?
Dado el uso de ISpVoice, la API para el Text-to-Speech, es necesario ejecutar la aplicación desde un sistema Windows.

El único componente a configurar para el uso de esta aplicación es la API de Google Gemini, la cual es usada para dar coherencia a las frases escritas por los pacientes en los tableros antes de ser reproducidas.

Para realizar la configuración de esta API, se deben seguir estos pasos:

- Obtener una clave API: Se puede obtener a traves del siguiente enlace: https://aistudio.google.com/app/apikey
- Establecer una variable de entorno:
    - Abre el menú de inicio y busca "variables de entorno"
    - Selecciona "Editar las variables de entorno del sistema"
    - En la ventana de "Propiedades del sistema", haz clic en "Variables de entorno"
    - En la sección "Variables de usuario", haz clic en "Nueva" para agreagar la nueva variable de entorno.
    - Introduce el nombre : "GOOGLE_API_KEY"
    - En el campo del valor, introduce la clave que has obtenido del enlace comentado anteriormente.
- Una vez realizados estos pasos, puede ser necesario un reinicio del sistema para aplicar los cambios.


Para editar los tableros -> 

Para el uso de Google Gemini como conjugador de frases automático es necesaria la API configurada y tener conexión a internet. Si no se dispone de esto, las frases sonarán pero no serán conjugadas automáticamente.


## ¿Cómo ejecutar la aplicación?
Se está trasladando a Docker, pero por el momento:
- pip install -r requirements.txt
- python3 main.py

