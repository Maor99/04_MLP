import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

class MLP:
    def __init__(self, layers_sizes):
        self.layers_sizes = layers_sizes

        self.weights = []
        self.biases = []
        self.activations_a = []

        for i in range(len(layers_sizes)-1):
            layers_weights = np.random.rand(layers_sizes[i], layers_sizes[i+1])-0.5

            layer_biases = np.random.rand(1, layers_sizes[i+1]) - 0.5

            self.weights.append(layers_weights)
            self.biases.append(layer_biases)

    def feedforward(self, inputs):
        activation = inputs
        self.activations_a = [inputs]

        for layer_weights, layer_bias in zip(self.weights, self.biases):
            z = np.dot(activation, layer_weights) + layer_bias
            activation = self.sigmoid(z)
            self.activations_a.append(activation)
        
        return activation  

    def sigmoid (self,x):
        return 1 / (1 + np.exp(-x))
    
    def train(self, inputs, ground_truth, learning_rate, iterations):
        for _ in range(iterations):
            self.backpropagation(inputs, ground_truth, learning_rate)
    
    def backpropagation(self, inputs, Y_ground_truth, learning_rate):
    # Feedforward para obtener las activaciones fainales (la salida)
        Y_hat_output = self.feedforward(inputs)
    
    # Cálculo de la cantidad de ejemplos (aproximación de "mini-batch")
        n = Y_ground_truth.shape[0]
    
    # Lista para almacenar los deltas de cada capa
        delta_capa = [None] * len(self.weights)
    
    # Hay dos calculos (etapas) de la regla de la cadena que se pasan 
    # de capa a capa (asì se retro-propaga el error).        
            
    # La capa de salida es distinta, porque implica la derivada de la función de COSTO = d_MSE.
    # d_C /d_O: Detal de la función de salida respecto a la función de activación (sigmoide)
        delta_A = self.derive_C_respect_derive_A(Y_ground_truth, Y_hat_output)
    
    # d_O / d_Z: Detal de la función de activación (sigmoide) respecto a la sumatoria 
    # // Derivada de la activación respecto a las zumatorias // es la derivada de la funcion Sigmoide.
        delta_Z = self.derive_A_respec_derive_Z(Y_hat_output)
    
    # Acumulación de los dos deltas anteriores. En el caso de la capa de salida: 
    # L[-1] es la acumulación de d_C / d_O * d_O / d_Z:n
        delta_capa[-1] = delta_A * delta_Z
    
    # Retropropagación del error a las capas ocultas...
        for l in range(len(delta_capa) - 2, -1, -1):
            delta_A = self.derive_E_respect_derive_A(delta_capa, l) 
            delta_Z = self.derive_A_respec_derive_Z(self.activations_a[l+1])
            delta_capa[l] = delta_A * delta_Z
    
    # Ajuste de pesos y sesgos
        for l in range(len(self.weights)):
            # D_Z / D_W:
            delta_W = self.derive_Z_respect_derive_W(delta_capa, l)
            self.weights[l] = self.weights[l] - learning_rate * delta_W
            # D_Z / D_B:
            delta_B =  self.derive_Z_respect_derive_B(delta_capa, l)
            self.biases[l] = self.biases[l] -  learning_rate * delta_B            
            

    def derive_C_respect_derive_A(self, Y_ground_truth, Y_hat_output):
        """
        El valor de La Derivada de la Función de Costo MSE (Media de los Errores al Cuadrado), 
        Utiliza como datos a Y_ground_truth [ (y) ] y a Y_hat_output [ (y^) -o- f(x) ].
        MSE = 1/n __SUM(desde i=1 hasta i=n)__ (y_i - f(x)_i)^2
        su derivada se vuelve d_MSE/d_x = -2(y - f(x))
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        NOTA: Otra función de costo, para no utilizar el 2 ni el negativo es SE (Errores al Cuadrado) con el truco de 1/2.
                SE = 1/2 (y - f(x))^2; la derivada se vuelve d_SE/d_x = (fx) - y

        Args:
            Y_ground_truth (_type_): _description_
            Y_hat_output (_type_): _description_

        Returns:
            _type_: _description_
        """
        return -2 * (Y_ground_truth - Y_hat_output) # / n (de alguna forma lo "sopesa" el learling rate)


    def derive_E_respect_derive_A(self, delta_capa, l):
        """
        -> d_C/d_O * d_A/d_Z * d_Z/d_A
        En las capas ocultas, la derivada del 'Error' es el valor acumulado del error en 
        las derivadas parciales y las derivadas concatenadas (por la regla de la cadena), 
        que llegan desde la capa de salida (o en caso de varias capas ocultas, acumuladas también desde estas)
        
        -> "d_C/d_O" y "d_A/d_Z" ya se tienen, está contenido dentro de "delta_capa"
        
        -> d_Z/d_A da como resultado la capa de pesos, puesto que todas las As se vuelven cero,
        a excepción de la A_t que al no tener exponente su derivada da 1, y se multiplica por el valor de W que se vuelve constante.!
        Z = w_1 * a_1 + w2 * a_2 ... + wn * a_n
        .:. d_Z / d_A = W de la capa en cuestión.
        .:. Práctaimente el resultado es la capa de pesos (en su peso) 
            multiplicado por el pequeño factor de error que se viene arrastrando desde la salida (BACK-PROP de capa en capa)
            ... esto es así, por lo que el calculo es independiente por capa (por algoritmo -> por MATRICES : Paralelismo -> GPU)
            ... esta parte se ejecuta secuencialemnte (hacia atrás: BACK) solo luego de haber ejecutado toda la capa posterior primero.
            ... en la resolución análitica, esto no se aprecia porque es una sola cadena (no por capa)
        """
        # la W se transpone, de tal forma que W.T se vuelva de columnas (1 x n).
        # delta_capa es de filas (m x 1)... del producto cruz surge una matriz de m x n
        return np.dot(delta_capa[l+1], self.weights[l+1].T)
        # Para las capas


    def derive_A_respec_derive_Z(self, x):
        """
        La derivada de la función de activacón utilizada es la derivada de la función sigmoide,
        esta se escribe en función a la misma función_sigmoide.
        En este caso los valores de Xs representan la salida de la función_sigmoide.
        La clase MLP almacena las salidas de la función sigmoide en self.activations_a.
        A la vez, la entrada a la función sigmoide fue la salida de la función de sumatoria Z.
        """
        return x * (1 - x)


    def derive_Z_respect_derive_W(self, delta_capa, l):
        """
        -> d_C/d_O * d_A/d_Z * d_Z/d_W
        En este punto se tiene acumulado el valor del error debido a  
        las derivadas parciales y las derivadas concatenadas (por la regla de la cadena),
        que llegan desde la capa de salida (o en caso de varias capas ocultas, acumuladas también desde estas)
        
        -> "d_C/d_O" y "d_A/d_Z" ya se tienen, está contenido dentro de "delta_capa"
        
        ->  d_Z/d_WA da como resultado la salida de las activaciones, puesto que todas las Ws se vuelven cero,
        a excepción de la W_t que al no tener exponente su derivada da 1, y se multiplica por el valor de A que se vuelve constante!
        Z = w_1 * a_1 + w2 * a_2 ... + wn * a_n
        .:. d_Z / d_W = A de la capa en cuestión.
        """
        return np.dot(self.activations_a[l].T, delta_capa[l])


    def derive_Z_respect_derive_B(self, delta_capa, l):
        return np.sum(delta_capa[l], axis=0, keepdims=True)


class Plotter:
    def __init__(self, casas_salidas, predicciones):
        self.casas_salidas = np.array(casas_salidas)
        self.predicciones = np.array(predicciones)

    def plot_error_distribution(self):
        mse = mean_squared_error(self.casas_salidas, self.predicciones)
        rmse = sqrt(mse)
        plt.figure(figsize=(10, 6))
        plt.hist(self.predicciones - self.casas_salidas, bins=20, alpha=0.7, label='Error de Predicción')
        plt.axvline(x=0, color='r', linestyle='--')
        plt.title(f'MSE: {mse:.2f} || RMSE: {rmse:.2f}')
        plt.xlabel('Error')
        plt.ylabel('Frecuencia')
        plt.legend()
        plt.show()

    def plot_prediction_errors(self):
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=self.casas_salidas.flatten().tolist(), y=self.predicciones.flatten().tolist(), color="blue", label="Errores de Predicción")
        sns.regplot(x=self.casas_salidas.flatten().tolist(), y=self.predicciones.flatten().tolist(), scatter=False, color="red", label="Tendencia e Intervalo de Confianza")
        plt.title('Errores de Predicción con Línea de Tendencia e Intervalo de Confianza')
        plt.xlabel('Índice de la Observación')
        plt.ylabel('Error de Predicción')
        plt.legend()
        plt.show()