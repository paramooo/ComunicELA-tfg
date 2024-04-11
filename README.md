# ComunicELA-tfg

Datos recopilacion:
[0 - 15] -> Distancias de los puntos de control del ojo izquierdo a la pupila izquierda (en pixeles)
[16 - 31] -> Distancias de los puntos de control del ojo izquierdo a la pupila derecha  (en pixeles)
[32] -> Inclinación de la cabeza sobre el eje X        (entre 0 y 1, max 45º para cada lado)
[33] -> Inclinación de la cabeza sobre el eje Y        (entre 0 y 1, max 45º para dada lado)
[34] -> Posicion de la cabeza en la camara en el eje X        (entre 0 y 1)
[35] -> Posicion de la cabeza en la camara en el eje Y        (entre 0 y 1)
[36] -> Medida EAR      (entre 0 y 1)
[37] -> Umbral EAR      (calibrado por la persona anteriormente)



normalizar los datos en base a esa distancia, (buscar alguna formula o algo nose)


Informarme sobre las redes secuenciales, capas LSTM o GRU
Ahora esta puesto suavizando la posicion reconocida con las posiciones reconocidas anteriormente pero vaya creo que asi ira mejor
El problema de hacer esto es que creo que para el entrenamiento van a hacer falta datos con la pelota moviendose aleatoriamente por la pantalla
No va a valer asi la secuencia de izquierda a derecha y de abajo a arriba


crear exe:
python -m PyInstaller --onefile --icon=imagenes/logo.ico --add-data "anns;./anns" --add-data "imagenes;./imagenes" --add-data "kivy;./kivy" --add-data "sonidos;./sonidos" main.py



falta pantalla completa

no poder entrar en los menus si no hay camara activa

incorporar posibilidad de guardar diferentes usuarios despues de calibrar