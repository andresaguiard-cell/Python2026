import numpy as np

def inicializar_parametros(dimensiones):
    """
    Inicializa pesos y sesgos con una semilla aleatoria fija.
    
    Args:
        dimensiones (int): Número de píxeles de la imagen aplanada.
    Returns:
        tuple: Matriz de pesos W y valor de sesgo b.
    """
    np.random.seed(42)
    W = np.random.randn(dimensiones, 1) * 0.01
    b = 0
    return W, b

def sigmoide(z):
    """Función de activación para la unidad lógica del modelo."""
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, W, b):
    """
    Realiza la composición funcional de la transformación lineal y activación.
    """
    z = np.dot(X, W) + b
    return sigmoide(z)
    