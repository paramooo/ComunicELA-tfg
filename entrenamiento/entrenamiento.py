# import cv2
# import numpy as np
# import tensorflow as tf
# from Conjuntos import Conjuntos
# #from provisionales import evaluar_zonas

# # Funcion para cargar los datos de entrenamiento
# def cargar_datos():
#     # Cargar los datos
#     input = np.loadtxt('./txts/input.txt', delimiter=',')
#     output = np.loadtxt('./txts/output.txt', delimiter=',')

#     return input, output

# # Funcion para cargar los datos de test
# def cargar_datos_test():
#     # Cargar los datos
#     input = np.loadtxt('./txts/input2.txt', delimiter=',')
#     output = np.loadtxt('./txts/output2.txt', delimiter=',')

#     return input, output

# # Funcion para aplicar un filtro gaussiano a las columnas de las distancias para suavizar el trazado de los datos de entrenamiento
# def gauss(data, sigma):
#     # Aplicar filtro gaussiano con cv2 a las columnas de las distancias (32)
#     for i in range(32):
#         data[:, i] = cv2.GaussianBlur(data[:, i], (1, sigma), 0).flatten()
#     return data


# # Funcion para crear la ANN
# def crear_ann(topology, learning_rate, loss_function):
#     model = tf.keras.models.Sequential()
#     model.add(tf.keras.layers.Input(shape=(topology[0],)))  # Entrada
#     for layer_size in topology[1:-1]:  # Capas ocultas
#         model.add(tf.keras.layers.Dense(layer_size, activation='relu'))
#     model.add(tf.keras.layers.Dense(topology[-1], activation='sigmoid'))  # Salida
#     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#     model.compile(loss=loss_function, optimizer=optimizer, metrics = [tf.keras.metrics.MeanSquaredError(),
#                                                                     tf.keras.metrics.MeanAbsoluteError(),
#                                                                     tf.keras.losses.LogCosh(),
#                                                                     tf.keras.metrics.MeanSquaredLogarithmicError(),
#                                                                     tf.keras.losses.Huber(),
#                                                                     euclidean_loss])
#     return model



# #Funcion de loss para evaluar con la distancia euclidea
# @tf.keras.utils.register_keras_serializable()
# def euclidean_loss(y_true, y_pred):
#     return tf.sqrt(tf.reduce_sum(tf.square(y_pred - y_true), axis=-1))


# # Funcion para entrenar la ANN
# def entrenar(topology, learning_rate, epochs, loss_function, input, output):
#     model = None
#     if loss_function == 'logcosh':
#             model = crear_ann(topology, learning_rate, tf.keras.losses.LogCosh())
#     elif loss_function == 'huber_loss':
#         model = crear_ann(topology, learning_rate, tf.keras.losses.Huber())
#     elif loss_function == 'euclidean_loss':
#         model = crear_ann(topology, learning_rate, euclidean_loss)
#     else:
#         model = crear_ann(topology, learning_rate, loss_function)

#     # Entrenar el modelo
#     model.fit(input, output, epochs=epochs, verbose=1)

#     return model



# # Funcion para evaluar con todas las metricas y guardar en un txt
# def evaluar_txt(model, input, output, model_name):
#     # Hacer la predicción con el modelo
#     predictions = model.predict(input)

#     # Evaluar el modelo
#     metrics = {
#         'mean_squared_error': np.mean(tf.keras.losses.MSE(output, predictions)),
#         'mean_absolute_error': np.mean(tf.keras.losses.MAE(output, predictions)),
#         'logcosh': np.mean(tf.keras.losses.log_cosh(output, predictions)),
#         'mean_squared_logarithmic_error': np.mean(tf.keras.losses.MSLE(output, predictions)),
#         'huber_loss': np.mean(tf.keras.losses.huber(output, predictions)),
#         'euclidean_loss': np.mean(euclidean_loss(output, predictions))
#     }

#     # Guardar los resultados en un txt
#     with open('./anns/results.txt', 'a') as f:
#         f.write(f"Modelo: {model_name}\n")
#         for name, value in metrics.items():
#             f.write(f"{name}: {value}\n")
#         f.write("\n")




# def entrenar_bucle(conjuntos, topologias, learning_rates, epochs, loss_functions):
#     # Cargar los datos
#     input, output = cargar_datos()
#     input_t, output_t = cargar_datos_test()

#     total_anns = len(conjuntos) * len(topologias) * len(learning_rates) * len(epochs) * len(loss_functions)
#     current_ann = 0

#     for conjunto in conjuntos:

#         # Normalizar los datos para el conjunto
#         normalizar_funcion = getattr(Conjuntos, f'conjunto_{conjunto}')
#         input_norm = normalizar_funcion(input)
#         input_test_norm = normalizar_funcion(input_t)

#         # Entrenar todas las combinaciones posibles
#         for i, topologia in enumerate(topologias[conjunto-1]):
#             for j, lr in enumerate(learning_rates):
#                 for k, ep in enumerate(epochs):
#                     for loss_function in loss_functions:
#                         current_ann += 1
#                         print(f"Entrenando conjunto:{conjunto} topologia {topologia} lr:{lr} ep:{ep} loss:{loss_function} ({current_ann}/{total_anns})")
#                         model = entrenar(topologia, lr, ep, loss_function, input_norm, output)

#                         # Guardar el modelo
#                         model_name = f"conj{conjunto}_{'_'.join(map(str, topologia))}_Lr-{lr}_Epo-{ep}_Loss-{loss_function}"
#                         #model.save(f'./anns/{model_name}.keras')

#                         # Evaluar el modelo e ir añadiendo los resultados al txt
#                         evaluar_txt(model, input_test_norm, output_t, model_name)



# def main():
#     # Definimos las topologias que vamos a probar
#     conjuntos = [3]
#     topologias= [#Las del conjunto 1
#                     []
#                     #Las del conjunto 2
#                     ,[]
#                     #Las del conjunto 3
#                     ,[[22, 20, 15, 4, 2], [22, 15, 10, 2],  [22, 10, 6, 2], [22, 12, 2], [22, 20, 2]]]
    
#     learning_rates = [0.1, 0.05, 0.01, 0.005, 0.001]
#     epochs = [100, 200, 500, 1000]
#     loss_functions = ['mean_squared_error', 'mean_absolute_error', 'logcosh', 'mean_squared_logarithmic_error', 'huber_loss', 'euclidean_loss']

#     # Entrenar las topologias
#     entrenar_bucle(conjuntos, topologias, learning_rates, epochs, loss_functions)


# if __name__ == "__main__":
#     main()

