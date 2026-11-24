from pathlib import Path
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
import os
from PIL import Image

bmp_path = RAIZ_PROJETO / "dados/imagens_originais/Pulseiris03filtro09W000000.bmp"
cma_path = RAIZ_PROJETO / "dados/imagens_originais/Pulseiris03filtro09W000000.cma"

bmp = Image.open(bmp_path)
largura, altura = bmp.size

n_pixels = largura * altura
tamanho_cma = os.path.getsize(cma_path)

print("Dimensões:", largura, "x", altura)
print("Pixels:", n_pixels)
print("Tamanho CMA:", tamanho_cma, bytes)

esperado_16bits = n_pixels * 2
esperado_10bits_compactado = (n_pixels * 10 + 7) // 8

print("Esperado se for 16 bits por pixel:", esperado_16bits)
print("Esperado se for 10 bits compactado:", esperado_10bits_compactado)

print("Possível cabeçalho no caso 16 bits:", tamanho_cma - esperado_16bits)
print("Possível cabeçalho no caso 10 bits compactado:", tamanho_cma - esperado_10bits_compactado)

import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

bmp_path = RAIZ_PROJETO / "dados/imagens_originais/Pulseiris03filtro09W000000.bmp"
cma_path = RAIZ_PROJETO / "dados/imagens_originais/Pulseiris03filtro09W000000.cma"

# Pega dimensões a partir do BMP
largura, altura = Image.open(bmp_path).size
n_pixels = largura * altura

print("Largura:", largura)
print("Altura:", altura)
print("Pixels esperados:", n_pixels)

# Lê o arquivo CMA como texto
with open(cma_path, "r", encoding="ascii", errors="ignore") as f:
    texto = f.read()

# Extrai todos os números do arquivo
numeros = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", texto)

valores = np.array([float(n) for n in numeros], dtype=np.float32)

print("Quantidade de números encontrados:", valores.size)
print("Mínimo geral:", valores.min())
print("Máximo geral:", valores.max())
print("Valores por imagem:", n_pixels)
print("Quantidade aproximada de imagens:", valores.size / n_pixels)
print("Resto da divisão:", valores.size % n_pixels)