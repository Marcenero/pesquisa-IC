import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

cma_path = "Pulseiris03filtro09W000000.cma"

#Abre o arquivo .cma em modo texto
with open(cma_path, "r", encoding="ascii", errors="ignore") as f:
    texto = f.read()

#Encontra todos os números dentro do texto usando expressão regular
numeros = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", texto)

#Converte a lista de strings para um array NumPy de números reais
valores = np.array([float(n) for n in numeros], dtype=np.float32)

print("Quantidade de valores:", valores.size)
print("Mínimo:", valores.min())
print("Máximo:", valores.max())

#Com 1.000.000 de valores, assumi que a imagem tem dimensão 1000x1000
#Aqui reorganiza a lista de valores em uma matriz 2D
img = valores.reshape((1000, 1000))

#Mostra a imagem do arquivo .cma em tons de cinza
plt.imshow(img, cmap="gray")
plt.colorbar()
plt.title("Imagem CMA 1000 x 1000")
plt.show()

bmp_path = "Pulseiris03filtro09W000000.bmp"
bmp = Image.open(bmp_path)

#Pega a paleta do BMP
palette = np.array(bmp.getpalette(), dtype=np.uint8).reshape(-1, 3)

#Usa o maior índice realmente presente no BMP
bmp_indices = np.array(bmp)
idx_max = bmp_indices.max()

print("Maior índice de cor usado no BMP:", idx_max)

#Normaliza o CMA entre 0 e 960
img_norm = np.clip(img, 0, 960) / 960.0

#Converte os valores do CMA para índices da paleta
indices = (img_norm * idx_max).astype(np.uint8)

#Aplica a paleta
img_rgb = palette[indices]

#Salva a imagem colorida como .png
Image.fromarray(img_rgb).save("cma_colorida_com_paleta_bmp.png")

#Mostra a imagem .cma com a paleta de cores da imagem .bmp
plt.imshow(img_rgb)
plt.axis("off")
plt.title("CMA usando a paleta do BMP")
plt.show()