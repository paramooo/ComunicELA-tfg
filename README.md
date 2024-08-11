# ComunicELA: Software de Asistencia para la Comunicación en Pacientes con Esclerosis Lateral Amiotrófica

## Resúmen
La Esclerosis Lateral Amiotrófica (ELA) es una enfermedad neurodegenerativa que afecta
a un gran número de personas, deteriorando progresivamente las funciones motoras y, en
última instancia, afectando a la capacidad para hablar, tragar y respirar.
Uno de los desafíos importantes ocurre cuando la enfermedad progresa y la persona pierde
tanto la habilidad de hablar como de moverse. Las soluciones que tenemos hoy en día varían
dependiendo de cuánto pueda moverse la persona. Si todavía puede mover un poco, se usan
tableros para que pueda señalar con el dedo y comunicarse. Si solo puede mover la cabeza,
se utilizan sistemas que siguen este movimiento. Pero, cuando ya no puede mover nada, se
recurre a sistemas poco accesibles de seguimiento de la mirada.
Para abordar los desafíos asociados con la ELA, este proyecto propone desarrollar un siste-
ma que mejore significativamente la calidad de comunicación de las personas diagnosticadas.
La idea central consiste en crear una aplicación que sigua la posición de la vista del pacien-
te en la pantalla, facilitando diversas formas de comunicación a través de diferentes paneles
con palabras que serán utilizados mediante el movimiento de los ojos. Esta tecnología busca
proporcionar una herramienta accesible y eficaz para la comunicación de las personas que
padecen esta enfermedad o que sufren de falta de movimiento.

## Abstract
The Amyotrophic Lateral Sclerosis (ALS) is a neurodegenerative disease that affects a large
number of people, progressively deteriorating motor functions and ultimately affecting the
ability to speak, swallow, and breathe.
One of the significant challenges occurs when the disease progresses, and the person
loses both the ability to speak and move. The solutions we have today vary depending on
how much the person can move. If they can still move a bit, boards are used so they can point
with their finger and communicate. If they can only move their head, systems that follow
this movement are used. But when they can no longer move at all, eye-tracking systems are
resorted to, which tend to be quite costly.
To address the challenges associated with ALS, this end-of-degree project proposes to
develop a system that significantly improves the communication quality of diagnosed people.
The central idea is to create an application that follows the patient’s eye position on the screen,
facilitating various forms of communication through different panels with words that will be
used through eye movement. This technology seeks to provide an accessible and effective
tool for the communication of people suffering from this disease or who suffer from lack of
movement.
2


## ¿Qué es necesario configurar para su uso?
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


## Detalles sobre el desarrollo de ComunicELA
Este software es un proyecto de fin de grado aún en desarrollo. Debido a esto, el software no tiene implementados los últimos modelos investigados para el reconocimiento de la miradada.

### Backend:
No terminado al completo, el modelo que utiliza actualmente para la predicción de la mirada tiene una desviación de aproximadamente un 20% del punto a donde realmente el usuario esta mirando.

Cabe destacar que ya se han hecho avances en este aspecto llegando a un 8'3% de precisión pero debido a que la investigación sigue aún en desarrollo, no se han implementado estas mejoras.

El desarrollo de los modelos de redes neuronales están en la carpeta /entrenamiento, aunque como ya se ha comentado, esto es una versión preeliminar que aún continua en desarrollo.

