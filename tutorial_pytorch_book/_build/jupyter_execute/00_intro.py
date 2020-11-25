#!/usr/bin/env python
# coding: utf-8

# # Introducción 
# ## ¿Qué es el Deep Learning? 
# Según la [Wikipedia](https://es.wikipedia.org/wiki/Aprendizaje_profundo), el *Deep Learning* o **aprendizaje profundo** es un conjunto de algoritmos de aprendizaje automático que intenta modelar abstracciones de alto nivel en datos utilizando arquitecturas computacionales que admiten transformaciones no lineales múltiples e iterativas de datos expresados en forma matricial o tensorial. 
# 
# Esto se realiza mediante el uso de [redes neuronales](https://es.wikipedia.org/wiki/Red_neuronal_artificial) de múltiples capas en una grandísima variedad de arquitecturas. Podéis ver una charla sobre *deep learning*, redes neuronales y diferentes arquitecturas en este canal de youtube: 

# In[1]:


from IPython.display import YouTubeVideo, HTML
YouTubeVideo('KqRnYGVjSNs', width=800, height=400)
# Ir a la URL: https://www.youtube.com/watch?v=KqRnYGVjSNs


# O una versión rápida de dicha introducción en 3 minutos:

# In[2]:


YouTubeVideo('4xaPxNPr43w', width=800, height=400)
# Ir a la URL: https://www.youtube.com/watch?v=4xaPxNPr43w


# ## Preparación del material
# Para la realización de la práctica vamos a utilizar los siguientes **recursos**: 
# * Lenguaje de programación **Python 3** (recomendada la distribución [Anaconda](https://www.anaconda.com/distribution/))
# * Formato de archivos: [Jupyter notebook](https://jupyter.org) 
# * Librería [PyTorch](https://pytorch.org/get-started/locally/)
# * Bases de datos de imágenes MNIST handwritten digit database
# 
# En las siguientes secciones veremos qué es y cómo se instalan dichos componentes.

# ![cropped-Anaconda_horizontal_RGB-1-600x102.png](attachment:cropped-Anaconda_horizontal_RGB-1-600x102.png)

# ### Distribución Anaconda
# [Anaconda](https://www.anaconda.com/distribution/) es una distribución del lenguaje de programación Python, orientada a la ciencia e ingeniería. Dispone de numerosas librerías que facilitan el trabajo como `numpy` (computación con arrays), `scipy` (estadística y procesado de señal) o `scikit-learn` (machine learning), entre otras muchas, convirtiéndola en la mayor **rival de Matlab**. Ésto, unido a que es **software libre** han hecho que su popularidad se haya visto incrementada en los últimos años, llegando a ser el lenguaje de programación más popular, incluso por encima de Java:

# ![Captura%20de%20pantalla%20de%202020-04-23%2009-09-40.png](attachment:Captura%20de%20pantalla%20de%202020-04-23%2009-09-40.png)

# Fijáos en como la popularidad de matlab decrece conforme aumenta la de python, porque están muy relacionadas. 
# 
# Para la instalación de Anaconda nos vamos a su [página oficial](https://www.anaconda.com/distribution/) o directamente lo descargamos en su versión para:
# * Windows [Descarga](https://repo.anaconda.com/archive/Anaconda3-2020.02-Windows-x86_64.exe)
# * MacOSX [Descarga](https://repo.anaconda.com/archive/Anaconda3-2020.02-MacOSX-x86_64.pkg)
# * Linux [Descarga](https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh)
# 
# Una vez descargado, realizamos la instalación de dicho paquete. Se detallan a continuación los pasos en cada sistema operativo. 
# #### Windows
# 1. Ejecutar el archivo y "siguiente" 
# 2. Leer la licencia y click en "I Agree". 
# 3. Selecciona una instalación "Solo para mi" (a menos que vayas a compartir el ordenador con otros usuarios). 
# 4. Selecciona una carpeta de destino para instalar Anaconda y haz clic en el botón Siguiente.  
# 5. Elegir si desea añadir Anaconda a su variable de entorno PATH. Se recomienda **no** añadir Anaconda a la variable de entorno del PATH, ya que puede interferir con otros programas y en su lugar utilizarla abriendo el "Anaconda Navigator" desde el menú de inicio.
# 6. Elegir si desea registrar Anaconda como Python predeterminado. A menos que tengas instaladas o quieras ejecutar varias versiones de Anaconda o varias versiones de Python, dejar la casilla predeterminada marcada. 
# 7. Haz clic en el botón Instalar. Si quieres ver los paquetes que Anaconda está instalando, haz clic en Mostrar detalles.
# 8. Haga clic en el botón Siguiente.
# 9. Os ofrecerá instalar PyCharm, un IDE para desarrollo en python. Nosotros no lo vamos a utilizar, así que si no queréis o no sabéis que es, click en "Siguiente". 
# 10. Después de una instalación exitosa verá el cuadro de diálogo "Gracias por instalar Anaconda"
# 
# #### MacOS
# 1. Haga doble clic en el archivo descargado y haga clic en continuar para iniciar la instalación.
# 2. Responda a las preguntas de las pantallas de Introducción, Léame y Licencia.
# 3. Haga clic en el botón Instalar para instalar Anaconda en su directorio ~/opt (recomendado):
# 4. En la pantalla de selección de destino, seleccione Instalar sólo para mí.
#     * Si aparece el mensaje de error "No puede instalar Anaconda en esta ubicación", vuelva a seleccionar Instalar sólo para mí.
# 5. Haga clic en el botón de continuar.
# 6. Os ofrecerá instalar PyCharm, un IDE para desarrollo en python. Nosotros no lo vamos a utilizar, así que si no queréis o no sabéis que es, click en "Siguiente". 
# 7. Después de una instalación exitosa verá el cuadro de diálogo "Gracias por instalar Anaconda"
# 
# #### Linux
# 1. Abre la temrinal y navega hasta la carpeta de descargas
# 2. Introduce: `bash ~/Descargas/Anaconda3-2020.02-Linux-x86_64.sh` o el nombre del archivo descargado. Incluya el comando bash sin importar si está usando o no el shell Bash.
# 3. El instalador le pide "Para continuar el proceso de instalación, por favor revise el acuerdo de licencia". Haz clic en Intro para ver los términos de la licencia. Desplácese hasta la parte inferior de los términos de la licencia e introduzca "Sí" para aceptar.
# 4. El instalador le pedirá que haga clic en Intro para aceptar la ubicación de instalación predeterminada, en `/home/<usuario>/anaconda3`
# 5. El instalador pregunta: "¿Desea que el instalador inicialice Anaconda3 ejecutando conda init?" Recomendamos "sí". Si por error respondes no, puedes añadir anaconda al path con el paso siguiente:
#     * primero ejecute `source /home/<usuario>/anaconda3/bin/activate` y luego ejecute `conda init`
# 6. El instalador termina y muestra "¡Gracias por instalar Anaconda3!"
# 
# #### Verificar la instalación de Anaconda
# * Windows: Haga clic en Inicio, busque o seleccione Anaconda Navigator en el menú.
# * macOS: Haga clic en el Launchpad, seleccione el Anaconda Navigator. O, usa Cmd+Espacio para abrir Búsqueda y escribe "Navegador" para abrir el programa.
# * Linux: Abre la terminal y escribe `conda list`, que listará todos los paquetes de Anaconda. 
# 
# Para **solución de problemas**, consultar la [documentación de Anaconda](https://docs.anaconda.com/anaconda/install/). 

# ![nav_logo.svg](attachment:nav_logo.svg)

# ### Jupyter Notebook
# El **Jupyter Notebook** es una aplicación web de código abierto que permite crear y compartir documentos que contienen código en vivo, ecuaciones, visualizaciones y texto narrativo. Sus usos incluyen: limpieza y transformación de datos, simulación numérica, modelado estadístico, visualización de datos, aprendizaje automático y mucho más.
# 
# **Jupyter** se instala por defecto con Anaconda, de modo que solo es necesario lanzarlo. 
# 
# #### Abrir Jupyter
# * Windows: 
#     1. Haga clic en Inicio, lance Anaconda Navigator en el menú. 
#     2. Seleccione "Jupyter. 
# * macOS: 
#     1. Haga clic en el Launchpad, seleccione el Anaconda Navigator.
#     2. Lanza "Jupyter notebook"
# * Linux: 
#     1. Abre la terminal
#     2. Escribe `jupyter notebook`
#     
# En todos los casos se abrirá una ventana similar a la siguiente, en la que podrás navegar por el árbol de directorios y cargar los notebooks de ipython que quieras. Si quieres probar, navega hasta la carpeta donde hayas descargado esta práctica y selecciona el archivo `00 Practica Deep Learning - Introducción.ipynb`, y podrás ver este mismo fichero en su versión interactiva. 

# ![00_2_jupyter_new_notebook.png](attachment:00_2_jupyter_new_notebook.png)

# ![logo_pytorch.svg.png](attachment:logo_pytorch.svg.png)

# ### PyTorch
# **PyTorch** es un paquete de Python diseñado para realizar cálculos numéricos haciendo uso de la programación de tensores. Además permite su ejecución en GPU para acelerar los cálculos.
# 
# Normalmente PyTorch es usado tanto para sustituir numpy y procesar los cálculos en GPU como para la investigación y desarrollo en el campo del machine learning, centrado principalmente en el desarrollo de redes neuronales.
# 
# #### Alternativas a PyTorch
# En la actualidad disponemos de varias alternativas a PyTorch en su aplicación al machine learning, algunas de las más conocidas son:
# * **Tensorflow**: fue desarrollado por Google Brain Team. Es software libre diseñado para computación numérica mediante grafos.
# * **Theano**: es una librería de python que te permite definir, optimizar y evaluar expresiones matemáticas que implican cálculos con arrays multidimensionales de forma eficiente.
# * **Keras**: es una API de alto nivel para el desarrollo de redes neuronales escrita en Python. Utiliza otras librerías de forma interna como son Tensorflow, CNTK y Theano. Fue desarrollado con el propósito de facilitar y agilizar el desarrollo y la experimentación con redes neuronales. Hoy día se distribuye junto con **Tensorflow**.
# 
# Mediante un análisis de las tendencias en Google Trends, se puede observar que PyTorch no deja de crecer, mientras que Keras y Tensorflow llevan unos meses a la baja. Aún así, Tensorflow sigue siendo el framework más popular, pero desde mi punto de vista es más complejo de dominar, y depende mucho de su API, mientras que la programación en PyTorch es más intuitiva, por lo que es la librería elegida para esta práctica. 

# ![Captura%20de%20pantalla%20de%202020-04-23%2009-48-37.png](attachment:Captura%20de%20pantalla%20de%202020-04-23%2009-48-37.png)

# #### Instalación de Pytorch
# Para la instalación de pytorch, tenemos que lanzar la consola de **Anaconda**: 
# 
# * Windows: Haga clic en Inicio, busque o seleccione Anaconda Prompt en el menú.
# * MacOS y Linux: Abrir terminal. 
# 
# Ahora ve a la [página de instalación de pytorch](https://pytorch.org/get-started/locally/) y selecciona las características de tu instalación, que serán probablemente las que aparezcan por defecto:

# ![Captura%20de%20pantalla%20de%202020-04-23%2009-59-00.png](attachment:Captura%20de%20pantalla%20de%202020-04-23%2009-59-00.png)

# En PyTorch Build usaremos la version *Stable*, en el Package pondremos *Conda* para instalarlo con anaconda, y Language será *Python*. En cuanto al OS seleccionar Linux, Mac o Windows en función de vuestro sistema operativo y finalmente en CUDA pondremos *None*, a menos que tengáis una tarjeta gráfica de Nvidia (en ese caso es más complejo puesto que hay que instalar las librerías CUDA y CuDNN, así como los drivers de Nvidia). 
# 
# Copiar el texto que aparece al lado de "Run this Command" y pegar en la terminal o el "Anaconda Prompt" en windows, dar siguiente y ya estará instalado. 
# 
# #### Verificar la instalación
# Para verificar si se ha instalado, abrimos una terminal de python: 
# * Windows y MacOS: Abre Anaconda Navigator y lanza `qtconsole`. 
# * Linux: Abre la terminal y escribe `ipython`. 
# 
# Y en la consola de Python escribimos: 
# ```python
# from __future__ import print_function
# import torch
# x = torch.rand(5, 3)
# print(x)
# ```
# Si todo funciona correctamente, ya tenemos todos los requisitos necesarios para realizar la práctica. 
