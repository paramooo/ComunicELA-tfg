# ComunicELA-tfg

Datos recopilacion:
[0 - 15] -> Distancias de los puntos de control del ojo izquierdo a la pupila izquierda (en pixeles)
[16 - 31] -> Distancias de los puntos de control del ojo izquierdo a la pupila derecha  (en pixeles)
[32] -> Inclinación de la cabeza sobre el eje X  - arriba o abajo  (entre 0 y 1, max 45º para cada lado)
[33] -> Inclinación de la cabeza sobre el eje Y   - lados          (entre 0 y 1, max 45º para cada lado)
[34] -> Inclinación de la cabeza sobre el eje Z    - hombros       (entre 0 y 1, max 45º para cada lado)
[35] -> Posicion de la cabeza en la camara en el eje X        (entre 0 y 1)
[36] -> Posicion de la cabeza en la camara en el eje Y        (entre 0 y 1)
[37] -> Medida EAR      (entre 0 y 1)
[38] -> Umbral EAR      (calibrado por la persona anteriormente)




Informarme sobre las redes secuenciales, capas LSTM o GRU
Ahora esta puesto suavizando la posicion reconocida con las posiciones reconocidas anteriormente pero vaya creo que asi ira mejor
El problema de hacer esto es que creo que para el entrenamiento van a hacer falta datos con la pelota moviendose aleatoriamente por la pantalla
No va a valer asi la secuencia de izquierda a derecha y de abajo a arriba


crear exe:
python -m PyInstaller --onefile --icon=imagenes/logo.ico --add-data "anns;./anns" --add-data "imagenes;./imagenes" --add-data "kivy;./kivy" --add-data "sonidos;./sonidos" main.py


no poder entrar en los menus si no hay camara activa

incorporar posibilidad de guardar diferentes usuarios despues de calibrar



pytorch2.2.2
cuda12.1
cudnn8.9.7



una vez la persona hace la calibracion que opciones de personalizacion interesan? 
-Elegir el tipo de tablero




PONER COMO UMBRAL PREDETERMINADO AL FINAL LA MEDIA DE TODOS LOS UMBRALES DE TODA LA GENTE AL CALIBRAR
SUAVIZAR ORIENTACION PARA ENTENAR
MIRAR TIME ENTRE CLICK Y CLICK
MEDIR DESVIACION TIPICA MEDIA PARA LOS DIFERENTES CONJUNTOS DE DATOS

capa sigmoide para que quede el resultado entre 0 y 1
problema entre suavidad y retraso: mediana


poner explicaciones de los archivos arriba en comentarios y limpiar tdoo bien

fondomenus: https://unsplash.com/es/fotos/un-fondo-negro-con-un-diseno-ondulado-Cp4xHgvXt0M
fuentes: google_fonts
fRANCISCO -> TITULO
Orbitron -> Textos
fondo_frame_editado -> generado con IA (dalle-3)
fuente audio click -> (recortado)   https://assets.mixkit.co/active_storage/sfx/1133/1133.wav
fuente audio lock -> https://freesound.org/people/nebulasnails/sounds/405534/
fuente audio alarma -> https://freesound.org/people/Trancox/sounds/391905/


Combinar red para puntos con otra red para la imagen asociada

efinicientnet

hacer sistema de gestion y edicion de tableros


 1 -> reentrenar
 2 -> acabar calibrar
 3 -> hacer intro