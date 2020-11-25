# Iniciación a pyTorch

Este libro nace de unas prácticas diseñadas para la asignatura Tratamiento Digital de Voz e Imagen, del grado en Ingeniería de Sonido e Imagen por la Universidad de Málaga. 

## Organización de la práctica: 
Esta práctica consta de tres partes, cada una de ellas es un *notebook* de `jupyter`, o sea, un documento como el que estás visualizando en este momento. Las partes son las siguientes: 
* **Parte 0: Introducción al Deep Learning**. Es este *notebook*, preparatorio de la práctica mediante **realización guiada** en el que se tratarán:
    - Unas pinceladas a las matemáticas y la historia de las redes neuronales
    - Información sobre qué es y cómo instalar el software necesario para esta práctica: la distribución de python 3 [Anaconda](https://www.anaconda.com/distribution/), la librería de computación tensorial [PyTorch](https://pytorch.org/get-started/locally/) y la plataforma de notebooks `jupyter`. 
    - Una introducción a la computación matricial en python con `pytorch`, que en muchas cosas se parece a matlab. 
* **Parte 1: El Perceptrón Multicapa**. En esta parte se introduce el perceptrón multicapa (*Multi-Layer Perceptron* o MLP), la red neuronal más básica que podemos construir. Se verán detalladamente las pautas a seguir para crear, entrenar y evaluar una red neuronal utilizando la librería `pytorch`. Será una **realización guiada**, sin problemas a resolver. 
* **Parte 2: Redes Neuronales Convolucionales**. Esta parte será la **entregable** de la práctica de redes. En ella veremos las redes neuronales convolucionales, que son la **herramienta más potente** que existe a día de hoy para el procesado de imagen. Se utilizan en coches autónomos, en la búsqueda de imágenes de google, diagnóstico de enfermedades, interpretación del lenguaje, y muchas más aplicaciones. Veremos las particularidades de estas redes y cómo se implementan cada una. La segunda parte de este *notebook* será el **problema a resolver**: crear y entrenar una red convolucional para la detección de dígitos escritos a mano. 

La realización será autoguiada individual, siguiendo los *notebooks* que se proveen. Adicionalmente, el martes 19 a las 19:00 haremos una videotutoría para resolver dudas que puedan surgir con esta práctica. 