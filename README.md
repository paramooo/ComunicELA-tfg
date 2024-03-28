# ComunicELA-tfg

Datos recopilacion:
[0 - 15] -> Distancias de los puntos de control del ojo izquierdo a la pupila izquierda (en pixeles)
[16 - 31] -> Distancias de los puntos de control del ojo izquierdo a la pupila derecha  (en pixeles)
[32] -> Medida media de los dos ojos (media de todas las distancias de cada ojo)        (en pixeles)
[33] -> Orientacion de la cabeza sobre el eje X        (entre 0 y 1, max 45º para cada lado)
[34] -> Orientacion de la cabeza sobre el eje Y        (entre 0 y 1, max 45º para dada lado)
[35] -> Posicion de la cabeza en la camara en el eje X        (entre 0 y 1)
[36] -> Posicion de la cabeza en la camara en el eje Y        (entre 0 y 1)
[37] -> Medida EAR      (entre 0 y 1)
[38] -> Umbral EAR      (calibrado por la persona anteriormente)



Modificaciones pendientes:
columna 32 no la media de todas las medidas sino la distancia entre los puntos horizontales del ojo (2.2/2.5cm mas menos)

normalizar los datos en base a esa distancia, (buscar alguna formula o algo nose)


otro conjunto con menos puntos de cada ojo ya que son demasiados y no va




MODELO -> linea 200