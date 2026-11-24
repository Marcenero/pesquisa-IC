import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


RAIZ_PROJETO = Path(__file__).resolve().parents[2]
PASTA_IMAGENS = RAIZ_PROJETO / "dados/imagens_originais/camadasEscuraVerde"


def numero_camadas(nome):
    """
    Extrai o primeiro número do nome do arquivo.

    Exemplos:
    1_folha.png  -> 1
    5_folhas.png -> 5
    """
    resultado = re.search(r"\d+", nome)

    if resultado is None:
        raise ValueError(f"Não encontrei o número de folhas em {nome}")

    return int(resultado.group())


def estatisticas(canal):
    return {
        "media": np.mean(canal),
        "mediana": np.median(canal),
        "desvio": np.std(canal),
        "maximo": np.max(canal),
        "p99": np.percentile(canal, 99),
        "p999": np.percentile(canal, 99.9),
        "saturados_pct": np.mean(canal >= 255) * 100,
    }


# ---------------------------------------------------------
# 1. Encontrar todas as imagens
# ---------------------------------------------------------

arquivos = sorted(
    list(PASTA_IMAGENS.glob("*.png")),
    key=lambda arquivo: numero_camadas(arquivo.name)
)

if len(arquivos) == 0:
    raise RuntimeError("Nenhuma imagem encontrada.")


# ---------------------------------------------------------
# 2. Selecionar ROI usando uma imagem reduzida
# ---------------------------------------------------------

imagem_referencia = cv2.imread(str(arquivos[0]))

if imagem_referencia is None:
    raise RuntimeError("Não foi possível abrir a imagem de referência.")


# Tamanho máximo da imagem exibida na tela
LARGURA_MAX = 1000
ALTURA_MAX = 650

altura_original, largura_original = imagem_referencia.shape[:2]

# Calcula escala preservando proporção
escala = min(
    LARGURA_MAX / largura_original,
    ALTURA_MAX / altura_original,
    1.0
)

largura_preview = int(largura_original * escala)
altura_preview = int(altura_original * escala)

preview = cv2.resize(
    imagem_referencia,
    (largura_preview, altura_preview),
    interpolation=cv2.INTER_AREA
)


# ---------------------------------------------------------
# ROI DO LASER
# ---------------------------------------------------------

print("Selecione uma região DENTRO da mancha do laser.")

x_p, y_p, w_p, h_p = cv2.selectROI(
    "ROI LASER",
    preview,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("ROI LASER")


# Converter coordenadas do preview para imagem original
x = int(x_p / escala)
y = int(y_p / escala)
w = int(w_p / escala)
h = int(h_p / escala)


# ---------------------------------------------------------
# ROI DO FUNDO
# ---------------------------------------------------------

print("Selecione uma região de FUNDO no papel.")

xb_p, yb_p, wb_p, hb_p = cv2.selectROI(
    "ROI FUNDO",
    preview,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()


# Converter para coordenadas originais
xb = int(xb_p / escala)
yb = int(yb_p / escala)
wb = int(wb_p / escala)
hb = int(hb_p / escala)


print()
print(f"Escala de visualização: {escala:.3f}")
print("ROI laser original:", x, y, w, h)
print("ROI fundo original:", xb, yb, wb, hb)


# ---------------------------------------------------------
# 3. Processar cada quantidade de folhas
# ---------------------------------------------------------

resultados = []


for arquivo in arquivos:

    camadas = numero_camadas(arquivo.name)

    imagem_bgr = cv2.imread(str(arquivo))

    if imagem_bgr is None:
        print(f"Não foi possível carregar {arquivo}")
        continue

    # OpenCV carrega BGR.
    # Convertemos para RGB para evitar confusão.
    imagem = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

    roi_laser = imagem[
        y:y+h,
        x:x+w
    ].astype(np.float32)

    roi_fundo = imagem[
        yb:yb+hb,
        xb:xb+wb
    ].astype(np.float32)

    resultado = {
        "arquivo": arquivo.name,
        "camadas": camadas
    }

    canais = {
        "R": 0,
        "G": 1,
        "B": 2
    }

    for nome_canal, indice in canais.items():

        sinal = roi_laser[:, :, indice]
        fundo = roi_fundo[:, :, indice]

        estat_sinal = estatisticas(sinal)
        estat_fundo = estatisticas(fundo)

        # intensidade média da região iluminada
        resultado[f"{nome_canal}_media"] = estat_sinal["media"]

        # intensidade média do fundo
        resultado[f"{nome_canal}_fundo"] = estat_fundo["media"]

        # intensidade corrigida
        resultado[f"{nome_canal}_corrigido"] = (
            estat_sinal["media"] -
            estat_fundo["media"]
        )

        resultado[f"{nome_canal}_mediana"] = estat_sinal["mediana"]
        resultado[f"{nome_canal}_desvio"] = estat_sinal["desvio"]

        resultado[f"{nome_canal}_max"] = estat_sinal["maximo"]
        resultado[f"{nome_canal}_p99"] = estat_sinal["p99"]
        resultado[f"{nome_canal}_p999"] = estat_sinal["p999"]

        resultado[f"{nome_canal}_saturados_pct"] = (
            estat_sinal["saturados_pct"]
        )

    resultados.append(resultado)


# ---------------------------------------------------------
# 4. Criar DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(resultados)

df = df.sort_values("camadas").reset_index(drop=True)


# ---------------------------------------------------------
# 5. Normalizar em relação a 1 folha
# ---------------------------------------------------------

referencia = df.iloc[0]

for canal in ["R", "G", "B"]:

    valor_ref = referencia[f"{canal}_corrigido"]

    df[f"{canal}_relativo"] = (
        df[f"{canal}_corrigido"] /
        valor_ref
    )


# ---------------------------------------------------------
# 6. Mostrar resultados
# ---------------------------------------------------------

colunas = [
    "camadas",

    "R_corrigido",
    "G_corrigido",
    "B_corrigido",

    "R_relativo",
    "G_relativo",
    "B_relativo",

    "R_p999",
    "G_p999",
    "B_p999",

    "R_saturados_pct",
    "G_saturados_pct",
    "B_saturados_pct",
]

print()
print(df[colunas])


# ---------------------------------------------------------
# 7. Salvar CSV
# ---------------------------------------------------------

df.to_csv(
    RAIZ_PROJETO / "resultados/atenuacao/verde/resultados_intensidade.csv",
    index=False
)

# ---------------------------------------------------------
# 8. Gráfico do decaimento relativo
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

cores = {
    "R": "red",
    "G": "green",
    "B": "blue"
}

for canal in ["R", "G", "B"]:

    # Para o canal azul, adiciona barra de erro
    if canal == "G":

        valor_ref = referencia["G_corrigido"]

        desvio_relativo = (
            df["G_desvio"] /
            abs(valor_ref)
        )

        plt.errorbar(
            df["camadas"],
            df["G_relativo"],
            yerr=desvio_relativo,

            marker="o",
            linestyle="-",
            color="green",

            capsize=4,
            elinewidth=1.5,

            label="G"
        )

    # Para R e G, plota normalmente
    else:

        plt.plot(
            df["camadas"],
            df[f"{canal}_relativo"],
            marker="o",
            color=cores[canal],
            label=canal
        )


plt.xlabel("Número de folhas")
plt.ylabel("Intensidade relativa")
plt.title("Decaimento da intensidade com o número de folhas")

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
