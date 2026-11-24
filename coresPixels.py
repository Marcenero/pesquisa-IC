from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

imagem = Image.open("imagens/Montagem/fotoCamera4.png").convert("RGB")

pixels = np.array(imagem, dtype=np.float64)

R = pixels[:, :, 0]
G = pixels[:, :, 1]
B = pixels[:, :, 2]

soma = R + G + B

# Ignora pixels pretos
mascara = soma > 0

R = R[mascara]
G = G[mascara]
B = B[mascara]
soma = soma[mascara]

# Coordenadas cromáticas
r = R / soma
g = G / soma

# Histograma
bins = 100

contagem, r_edges, g_edges = np.histogram2d(
    r,
    g,
    bins=bins,
    range=[[0, 1], [0, 1]]
)

# Centros de cada célula
r_centros = (r_edges[:-1] + r_edges[1:]) / 2
g_centros = (g_edges[:-1] + g_edges[1:]) / 2

R_grid, G_grid = np.meshgrid(r_centros, g_centros)

# Região impossível: r + g > 1
contagem_plot = contagem.T.copy()
contagem_plot[R_grid + G_grid > 1] = np.nan

# Células sem nenhum pixel
contagem_plot[contagem_plot == 0] = np.nan

# Gráfico
plt.figure(figsize=(8, 7))

plt.imshow(
    contagem_plot,
    origin="lower",
    extent=[0, 1, 0, 1],
    aspect="equal"
)

# Delimitação r + g = 1
plt.plot(
    [0, 1],
    [1, 0],
    "--",
    linewidth=1
)

plt.xlabel("r")
plt.ylabel("g")
plt.title("Distribuição cromática dos pixels")

cbar = plt.colorbar()
cbar.set_label("Número de pixels")

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.show()