import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#1) Ler o arquivo .cma
cma_path = "Pulseiris03filtro09W000000.cma"

#Abrir o arquivo .cma como texto
with open(cma_path, "r", encoding="ascii", errors="ignore") as f:
    texto = f.read()

#ER para extrair os números do arquivo
numeros = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", texto)

#Converte os números de texto para float
valores = np.array([float(n) for n in numeros], dtype=np.float32)

print("Quantidade de valores:", valores.size)
print("Mínimo:", valores.min())
print("Máximo:", valores.max())

#Pelo resultado anterior, sabe-se que a imagem tem 1000 x 1000 valores.
altura = 1000  # eixo y
largura = 1000 # eixo z

if valores.size != altura * largura:
    raise ValueError("A quantidade de valores não corresponde a 1000 x 1000.")

# Transforma a lista 1D em matriz 2D.
# img[i, j]: altura = eixo y e largura = eixo z
img = valores.reshape((altura, largura))

#2) Corrigir fundo e preparar intensidade
#Para usar os valores como pesos de intensidade, é melhor remover o fundo e depois cortar valores negativos
#O percentil 5 será usado como estimativa simples do fundo, evitando que o ruído de fundo aumente a largura do feixe
fundo = np.percentile(img, 5)

img_corr = img - fundo

#Tudo que ficou negativo vira zero
#Assim, a variância é calculada apenas com intensidades positivas
img_corr = np.clip(img_corr, 0, None)

print("Fundo estimado:", fundo)
print("Mínimo corrigido:", img_corr.min())
print("Máximo corrigido:", img_corr.max())

#3. Calcular centro, sigma e diâmetro para cada posição z
#Vetor com as coordenadas y.
#Como ainda não se tem a escala física, y está em pixels.
y = np.arange(altura)

#Soma da intensidade em cada coluna z.
#soma_I[j] = soma das intensidades ao longo de y para a coluna j.
soma_I = np.sum(img_corr, axis=0)

#Evita divisão por zero caso alguma coluna tenha intensidade muito baixa.
soma_I_segura = np.where(soma_I == 0, np.nan, soma_I)

#Centroide do feixe em y para cada z:
#y0(z) = soma(y * I(y,z)) / soma(I(y,z))
y0 = np.sum(img_corr * y[:, None], axis=0) / soma_I_segura

#Variância em y para cada z:
#var(z) = soma(I(y,z) * (y - y0)^2) / soma(I(y,z))
variancia = np.sum(img_corr * (y[:, None] - y0[None, :])**2, axis=0) / soma_I_segura

# Sigma é a raiz da variância.
sigma = np.sqrt(variancia)

# Para uma gaussiana no formato:
# I(y) = I0 * exp(-(y-y0)^2 / (2*sigma^2))
#
# FWHM = 2*sqrt(2*ln(2))*sigma
diametro_fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma

# Para feixe gaussiano em óptica:
# I(y) = I0 * exp(-2*(y-y0)^2 / w^2)
#
# Nesse caso:
# w = 2*sigma
# diametro 1/e² = 2*w = 4*sigma
diametro_1e2 = 4 * sigma

# Eixo z, em pixels.
z = np.arange(largura)


# ============================================================
# 4. Mostrar imagem original em tons de cinza
# ============================================================

plt.figure()
plt.imshow(img, cmap="gray")
plt.colorbar(label="Intensidade")
plt.title("Imagem CMA original")
plt.xlabel("z")
plt.ylabel("y")
plt.show()


# ============================================================
# 5. Mostrar centro e largura sobre a imagem
# ============================================================

plt.figure()
plt.imshow(img, cmap="gray")
plt.colorbar(label="Intensidade")

# Linha central do feixe
plt.plot(z, y0, label="Centro y0")

# Bordas usando sigma
plt.plot(z, y0 + sigma, "--", label="y0 ± sigma")
plt.plot(z, y0 - sigma, "--")

# Bordas usando diâmetro 1/e².
# Como diametro_1e2 = 4*sigma, o raio 1/e² é 2*sigma.
plt.plot(z, y0 + 2*sigma, ":", label="Raio 1/e²")
plt.plot(z, y0 - 2*sigma, ":")

plt.title("Centro e largura do feixe ao longo de z")
plt.xlabel("z")
plt.ylabel("y")
plt.legend()
plt.show()


# ============================================================
# 6. Plotar o diâmetro em função de z
# ============================================================

plt.figure()
plt.plot(z, diametro_1e2, label="Diâmetro 1/e² = 4σ")
plt.plot(z, diametro_fwhm, label="FWHM = 2.355σ")

plt.title("Diâmetro do feixe ao longo da propagação")
plt.xlabel("z")
plt.ylabel("Diâmetro em pixels")
plt.legend()
plt.grid(True)
plt.show()


# ============================================================
# 7. Salvar os resultados em CSV
# ============================================================

# Cada linha do CSV terá:
# z, centro y0, sigma, FWHM, diâmetro 1/e²
resultado = np.column_stack([
    z,
    y0,
    sigma,
    diametro_fwhm,
    diametro_1e2
])

np.savetxt(
    "diametro_feixe_por_z.csv",
    resultado,
    delimiter=",",
    header="z_pixel,y0_pixel,sigma_pixel,diametro_fwhm_pixel,diametro_1e2_pixel",
    comments=""
)

print("Arquivo salvo: diametro_feixe_por_z.csv")