from pathlib import Path
RAIZ_PROJETO = Path(__file__).resolve().parents[2]
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(str(RAIZ_PROJETO / "dados/imagens_originais/Montagem/fotoCamera6.png"))

b, g, r = cv2.split(img)

linha = img.shape[0] // 2
coluna = img.shape[1] // 2

# Gráfico 1: varredura horizontal
plt.figure(figsize=(10, 4))

plt.plot(r[linha, :], label="R", color="red")
plt.plot(g[linha, :], label="G", color="green")
plt.plot(b[linha, :], label="B", color="blue")

plt.xlabel("Posição x [pixels]")
plt.ylabel("Intensidade")
plt.legend()
plt.grid()

plt.show()

# Gráfico 2: varredura vertical
plt.figure(figsize=(10, 4))

plt.plot(r[:, coluna], label="R", color="red")
plt.plot(g[:, coluna], label="G", color="green")
plt.plot(b[:, coluna], label="B", color="blue")

plt.xlabel("Posição y [pixels]")
plt.ylabel("Intensidade")
plt.legend()
plt.grid()

plt.show()