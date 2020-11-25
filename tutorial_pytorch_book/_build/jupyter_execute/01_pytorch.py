#!/usr/bin/env python
# coding: utf-8

# # Pytorch
# Como dijimos anteriormente, PyTorch es un paquete de Python diseñado para realizar cálculos numéricos haciendo uso de la programación de tensores. Además permite su ejecución en GPU para acelerar los cálculos.
# 
# En la práctica es un sustituto bastante potente de Numpy, una librería casi estándar para trabajar con arrays en python. 
# 
# ## ¿Cómo funciona pytorch? 
# Vamos a ver un tutorial rápido del tipo de datos de pytorch y cómo trabaja internamente esta librería. Para esto tendrás que haber seguido correctamente todos los pasos anteriores. Para esto necesitas la **versión interactiva del notebook**. 
# 
# Para esta sección: 
# * **Abre Jupyter** (consultar arriba)
# * Navega hasta el notebook `00 Práctica Deep Learning - Introducción.ipynb` y ábrelo.
# * Baja hasta esta sección. 
# 
# Pero antes de nada os cuento algunas diferencias entre matlab y python: 
# * Python es un **lenguaje de propósito general** mientras que matlab es un lenguaje **específico para ciencia e ingeniería**. Esto no es ni bueno ni malo; matlab es más fácil de utilizar para ingeniería sin preparación, pero python es más versátil. 
# * Debido a ello, **Matlab carga automáticamente todas las funciones** mientras que en Python, **hay que cargar las librerías que vamos a utilizar**. Esto hace que usar funciones en matlab sea más sencillo (dos letras menos que escribir), pero a costa de que es más difícil gestionar la memoria, y los nombres de funciones se puden superponer. Supon que `A` es una matriz. Para hacer la pseudoinversa, en matlab hacemos: 
# 
# ```matlab
# pinv(A)
# ```
# * en python tenemos que cargar la librería:
# ```python
# import scipy as sp
# sp.pinv(A)
# ```
# * Esto genera una cosa llamada **espacio de nombres**, en el que las funciones de cada librería van precedidas por su abreviatura (si importamos con `import x as y`) o el propio nombre si usamos `import torch`, `torch.tensor()`, mientras que en matlab basta con llamar a la función. Por ejemplo, cuando en matlab escribimos:
#     - `vector = [1, 2, 3]`
# * en python+pytorch necesitamos especificar que es un tensor (un array multidimensional):
#     - `vector = torch.tensor([1,2,3])`
# 
# Vamos a cargar la librería con `import torch` y ver que podemos, por ejemplo, construir una matriz de 5x3 aleatoria. Para ejecutar una celda, basta con seleccionarla (bien con las flechas del teclado, bien con el ratón) y pulsando `Ctrl+Enter` (o bien pulsando "Run" en la barra superior). 

# In[3]:


import torch
x = torch.rand(5, 3)
print(x)


# O una matriz de ceros:

# In[4]:


x = torch.zeros(5, 3, dtype=torch.long)
print(x)


# O a partir de unos datos dados, y podemos mostrarla con `print`, pero también acceder a sus características, como el tamaño de la matriz:

# In[5]:


x = torch.tensor([[5.5, 3, 3],[2,1, 5], [3,4,2],[7,6,5],[2,1,2]])
print(x)
print(x.shape)


# Con tensores se puede operar de forma normal:

# In[6]:


y = torch.rand(5, 3)
print(x + y)


# Pero OJO CUIDAO, tienen que ser del mismo tamaño, si no, va a dar error:

# In[7]:


y = torch.rand(2,3)
print(x+y)


# Se puede hacer *slicing* como en numpy o Matlab. Por ejemplo, para extraer la primera columna:

# In[8]:


print(x[:, 1])


# Otra característica que nos será de mucha utilidad es cambiar la forma de la matriz, que en otros lenguajes se conoce como `reshape`, y aquí es un método del objeto tensor llamado `view()`:

# In[9]:


x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())


# Podemos operar con tensores y valores escalares:

# In[10]:


y = x + 2
print(y)


# Y también podemos definir funciones que realicen estas operaciones que apliquemos a los diferentes tensores: 

# In[11]:


def modulo(x,y):
    aux = x**2 + y**2
    salida = torch.sqrt(aux)
    return salida

print(modulo(x,y))


# Y, una parte fundamental es que pytorch conserva memoria de las operaciones realizadas en un vector: 

# In[12]:


x = torch.ones(2, 2, requires_grad=True)
y = x + 2
print(y)


# La propiedad `grad_fn` será fundamental en el entrenamiento de redes neuronales, ya que guarda el gradiente de la operación o función que se haya aplicado a los datos. Esto se conserva a traves de todas las operaciones:

# In[13]:


z = y * y * 3
out = z.mean()

print(z, out)


# O incluso llevan cuenta de las operaciones realizadas con funciones:

# In[14]:


print(modulo(x,y))


# Para calcular el gradiente a lo largo de estas operaciones se utiliza la función `.backward()`, que realiza la propagación del gradiente hacia atrás. Podemos mostrar el gradiente $\frac{\partial out}{\partial x}$ con la propiedad `x.grad`, así que lo vemos: 

# In[15]:


out.backward()
print(x.grad)


# Habrá aquí una matriz de 2x2 con valores 4.5. Si llamamos el tensor de salida $o$, tenemos que:
# 
# $$ 
# o = \frac{1}{4} \sum_iz_i, \quad z_i = 3(x_i + 2)^2
# $$
# 
# Así que $z_i|_{x_i=1} = 27$. Entonces, la $\frac{\partial o}{\partial x_i} = \frac{3}{2}(x_i+2)$ y $\frac{\partial o}{\partial x_i} |_{x_i=1} = \frac{9}{2} = 4.5$
# 
# Gracias a esto, y a las matemáticas del algoritmo de propagación hacia atrás (*backpropagation*, ver video de introducción a la práctica), se pueden actualizar los pesos en función de una función de pérdida en las redes neuronales. Se puede activar y desactivar el cálculo del gradiente con la expresión `torch.no_grad()`. 

# In[16]:


print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)


# En la próxima sección, `01 Práctica Deep Learning - Perceptrón Multicapa.ipynb`, veremos como se construye y se entrena nuestra primera red neuronal utilizando estas características de pytorch. 
