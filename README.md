# ComunicELA: Software de Asistencia para la Comunicación en Pacientes con Esclerosis Lateral Amiotrófica

## Resumen
La Esclerosis Lateral Amiotrófica (ELA) es una enfermedad neurodegenerativa irreversible que causa una pérdida gradual de neuronas, deteriorando progresivamente las funciones motrices, así como la capacidad de hablar, tragar y respirar.

La calidad de vida de las personas con ELA que carecen de la capacidad de comunicarse mejora considerablemente si utilizan algún sistema de comunicación aumentativa y alternativa (CAA). Este proyecto propone desarrollar un sistema que mejore significativamente la calidad de comunicación de las personas con esta enfermedad.

La idea central consiste en crear una aplicación con una interfaz intuitiva, accesible, adaptable y sencilla, que permita la comunicación a través de tableros de comunicación personalizables, controlados mediante movimientos oculares y parpadeo, formando frases que pueden ser reproducidas mediante síntesis de voz.

## Abstract
Amyotrophic Lateral Sclerosis (ALS) is an irreversible neurodegenerative disease that causes a gradual loss of neurons, progressively deteriorating motor functions, as well as the ability to speak, swallow, or breathe.

The quality of life for people with ALS who lack the ability to communicate improves considerably if they use some form of augmentative and alternative communication (AAC) system. This project proposes developing a system that significantly improves the quality of communication for people with this disease.

The central idea is to create an application with an intuitive, accessible, adaptable, and simple interface, allowing communication through customizable communication boards controlled by eye movements and blinking, forming phrases that can be reproduced through voice synthesis.

## Requisitos
El único componente a configurar para el uso de esta aplicación es la API de Google Gemini, utilizada para dar coherencia a las frases escritas por los pacientes en los tableros antes de ser reproducidas. En caso de no configurarla, el conjugador automático no estará disponible.

### Configuración de la API de Google Gemini
1. **Obtener una clave API:** Obtener clave API
2. **Establecer una variable de entorno:**
    - Abre el menú de inicio de Windows y busca "variables de entorno".
    - Selecciona "Editar las variables de entorno del sistema".
    - En la ventana de "Propiedades del sistema", haz clic en "Variables de entorno".
    - En la sección "Variables de usuario", haz clic en "Nueva" para agregar la nueva variable de entorno.
    - Introduce el nombre: `GOOGLE_API_KEY`.
    - En el campo del valor, introduce la clave obtenida del enlace anterior.
3. **Reiniciar el sistema:** Puede ser necesario un reinicio del sistema para aplicar los cambios.

## Edición de Tableros

### Estructura de Carpetas
- **Carpeta "tableros":** Contiene los archivos de los tableros comunicativos.
- **Carpeta "pictogramas":** Contiene todas las fotografías utilizadas en las casillas con pictogramas. Estas fotografías son genéricas para todos los idiomas.

### Archivos de Tableros
- Los tableros son archivos con extensión `.xlsx`, cada uno nombrado según el idioma correspondiente.
- Dentro de cada archivo, los diferentes tableros se distribuyen en hojas. El primer tablero que aparece al abrir el archivo es el de la primera hoja, llamada `TAB. INICIAL`.

### Personalización de Casillas
- Cada casilla tiene dos columnas reservadas:
    - **Primera columna:** Nombre de la fotografía correspondiente de la carpeta de pictogramas.
    - **Segunda columna:** Nombre de la palabra a mostrar.
- Para personalizar una casilla, añade la fotografía con el nombre deseado a la carpeta de pictogramas y escribe este nombre en la primera columna reservada para la casilla que quieras modificar.

### Enlaces a Otros Tableros
- Si la palabra de la casilla comienza por `TAB. `, el software la reconoce como un enlace a otro tablero (otra hoja del archivo).
- Por ejemplo, si en la palabra de una casilla se escribe `TAB. RÁPIDO`, al pulsar esta casilla se abrirá el tablero correspondiente a la hoja `TAB. RÁPIDO`.

### Consejo Adicional
- Es recomendable incluir en cada tablero (excepto en el inicial) una casilla que permita volver al tablero inicial. Esto facilita la navegación entre los diferentes tableros.

## Ejecución de la Aplicación
Actualmente se está trasladando a Docker, pero por el momento puedes ejecutarla con los siguientes comandos:
```bash
pip install -r requirements.txt
python3 main.py
