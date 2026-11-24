
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


# ---------------------------------------------------------
# CONFIGURAÇÕES - LASER AZUL
# ---------------------------------------------------------

RAIZ_PROJETO = Path(__file__).resolve().parents[2]
(RAIZ_PROJETO / "resultados/atenuacao/azul").mkdir(parents=True, exist_ok=True)
PASTA_IMAGENS = RAIZ_PROJETO / "dados/imagens_originais/camadasEscuraUV"

LARGURA_MAX = 1000
ALTURA_MAX = 650


def numero_camadas(nome):
    """
    Extrai o primeiro número do nome do arquivo.

    Exemplos:
    1_folha.png  -> 1
    5_folhas.png -> 5
    """
    resultado = re.search(r"\d+", nome)

    if resultado is None:
        raise ValueError(
            f"Não encontrei o número de folhas em {nome}"
        )

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
# 2. Selecionar ROIs usando uma imagem reduzida
# ---------------------------------------------------------

imagem_referencia = cv2.imread(str(arquivos[0]))

if imagem_referencia is None:
    raise RuntimeError(
        "Não foi possível abrir a imagem de referência."
    )

altura_original, largura_original = imagem_referencia.shape[:2]

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
# 3. ROI DO LASER AZUL - RETANGULAR
# ---------------------------------------------------------

print("Selecione uma região dentro da mancha do laser azul.")

x_p, y_p, w_p, h_p = cv2.selectROI(
    "ROI LASER AZUL",
    preview,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyWindow("ROI LASER AZUL")

x = round(x_p / escala)
y = round(y_p / escala)
w = round(w_p / escala)
h = round(h_p / escala)


# ---------------------------------------------------------
# 4. ROI DO FUNDO - RETANGULAR
# ---------------------------------------------------------

print("Selecione uma região de fundo fora da mancha.")

xb_p, yb_p, wb_p, hb_p = cv2.selectROI(
    "ROI FUNDO",
    preview,
    showCrosshair=True,
    fromCenter=False
)

cv2.destroyAllWindows()

xb = round(xb_p / escala)
yb = round(yb_p / escala)
wb = round(wb_p / escala)
hb = round(hb_p / escala)

if w <= 0 or h <= 0 or wb <= 0 or hb <= 0:
    raise RuntimeError("Uma das ROIs não foi selecionada.")

print()
print(f"Escala de visualização: {escala:.3f}")
print("ROI laser:", x, y, w, h)
print("ROI fundo:", xb, yb, wb, hb)


# ---------------------------------------------------------
# 5. Processar cada imagem
# ---------------------------------------------------------

resultados = []

canais = {
    "R": 0,
    "G": 1,
    "B": 2
}

for arquivo in arquivos:

    camadas = numero_camadas(arquivo.name)

    imagem_bgr = cv2.imread(str(arquivo))

    if imagem_bgr is None:
        print(f"Não foi possível carregar {arquivo}")
        continue

    imagem = cv2.cvtColor(
        imagem_bgr,
        cv2.COLOR_BGR2RGB
    )

    if imagem.shape[:2] != (altura_original, largura_original):
        raise RuntimeError(
            f"Resolução diferente na imagem: {arquivo.name}"
        )

    # ROI retangular do laser - sem máscara
    roi_laser = imagem[
        y:y+h,
        x:x+w
    ].astype(np.float32)

    # ROI retangular do fundo
    roi_fundo = imagem[
        yb:yb+hb,
        xb:xb+wb
    ].astype(np.float32)

    resultado = {
        "arquivo": arquivo.name,
        "camadas": camadas
    }

    for nome_canal, indice in canais.items():

        sinal = roi_laser[:, :, indice]
        fundo = roi_fundo[:, :, indice]

        estat_sinal = estatisticas(sinal)
        estat_fundo = estatisticas(fundo)

        # Intensidade média da ROI iluminada
        resultado[f"{nome_canal}_media"] = (
            estat_sinal["media"]
        )

        # Intensidade média do fundo
        resultado[f"{nome_canal}_fundo"] = (
            estat_fundo["media"]
        )

        # Intensidade corrigida pelo fundo
        resultado[f"{nome_canal}_corrigido"] = (
            estat_sinal["media"] -
            estat_fundo["media"]
        )

        resultado[f"{nome_canal}_mediana"] = (
            estat_sinal["mediana"]
        )

        # Desvio padrão espacial dos pixels
        resultado[f"{nome_canal}_desvio"] = (
            estat_sinal["desvio"]
        )

        resultado[f"{nome_canal}_max"] = (
            estat_sinal["maximo"]
        )

        resultado[f"{nome_canal}_p99"] = (
            estat_sinal["p99"]
        )

        resultado[f"{nome_canal}_p999"] = (
            estat_sinal["p999"]
        )

        resultado[f"{nome_canal}_saturados_pct"] = (
            estat_sinal["saturados_pct"]
        )

    resultados.append(resultado)


# ---------------------------------------------------------
# 6. Criar DataFrame
# ---------------------------------------------------------

df = pd.DataFrame(resultados)

if df.empty:
    raise RuntimeError("Nenhuma imagem foi processada.")

df = df.sort_values("camadas").reset_index(drop=True)


# ---------------------------------------------------------
# 7. Normalizar em relação a 1 folha
# ---------------------------------------------------------

ref_linhas = df[df["camadas"] == 1]

if ref_linhas.empty:
    raise RuntimeError("Não foi encontrada imagem de 1 folha.")

referencia = ref_linhas.iloc[0]

for canal in ["R", "G", "B"]:

    valor_ref = referencia[f"{canal}_corrigido"]

    if abs(valor_ref) < 1e-6:
        print(
            f"Aviso: referência do canal {canal} "
            "muito próxima de zero."
        )

        df[f"{canal}_relativo"] = np.nan

    else:
        df[f"{canal}_relativo"] = (
            df[f"{canal}_corrigido"] /
            valor_ref
        )


# ---------------------------------------------------------
# 8. Mostrar resultados
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

print("\nRESULTADOS - LASER AZUL")
print(df[colunas].to_string(index=False))


# ---------------------------------------------------------
# 9. Salvar CSV
# ---------------------------------------------------------

df.to_csv(
    RAIZ_PROJETO / "resultados/atenuacao/azul/resultados_intensidade_azul.csv",
    index=False
)


# ---------------------------------------------------------
# 10. Gráfico do decaimento relativo
# ---------------------------------------------------------

plt.figure(figsize=(9, 6))

cores = {
    "R": "red",
    "G": "green",
    "B": "blue"
}

for canal in ["R", "G", "B"]:

    if canal == "B":

        # Barras de dispersão somente no canal azul
        valor_ref = referencia["B_corrigido"]

        if abs(valor_ref) < 1e-6:
            raise RuntimeError(
                "Referência azul muito próxima de zero."
            )

        desvio_relativo = (
            df["B_desvio"] /
            abs(valor_ref)
        )

        plt.errorbar(
            df["camadas"],
            df["B_relativo"],

            yerr=desvio_relativo,

            marker="o",
            linestyle="-",
            color="blue",

            capsize=4,
            elinewidth=1.5,

            label="B (desvio espacial)"
        )

    else:

        # Vermelho e verde sem barras
        plt.plot(
            df["camadas"],
            df[f"{canal}_relativo"],
            marker="o",
            color=cores[canal],
            label=canal
        )


plt.xlabel("Número de folhas")
plt.ylabel("Intensidade relativa")

plt.title(
    "Decaimento da intensidade - Laser azul"
)

plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
        RAIZ_PROJETO / "resultados/atenuacao/azul/decaimento_laser_azul.png",
    dpi=300
)

plt.show()


# ---------------------------------------------------------
# 11. Coeficiente efetivo de atenuação - Canal azul
# ---------------------------------------------------------

camadas = df["camadas"].to_numpy(dtype=float)

intensidades = df["B_relativo"].to_numpy(dtype=float)

# Considerar apenas intensidades positivas e finitas
mascara = (
    np.isfinite(intensidades) &
    (intensidades > 0)
)

x = camadas[mascara] - 1
y = np.log(intensidades[mascara])

if len(x) < 3 or np.sum(x**2) == 0:
    print("Dados insuficientes para o ajuste exponencial.")

else:
    # ln(Irel) = -k * (N - 1)
    k = -np.sum(x * y) / np.sum(x**2)

    # Modelo calculado
    x_modelo = np.linspace(
        camadas.min(),
        camadas.max(),
        200
    )

    y_modelo = np.exp(
        -k * (x_modelo - 1)
    )

    # R² calculado na escala original
    y_observado = intensidades[mascara]
    y_previsto = np.exp(-k * x)

    ss_res = np.sum((y_observado - y_previsto)**2)
    ss_tot = np.sum(
        (y_observado - np.mean(y_observado))**2
    )

    r2 = (
        1 - ss_res / ss_tot
        if ss_tot > 0
        else np.nan
    )

    print("\nCOEFICIENTE EFETIVO - LASER AZUL")
    print(f"k_B = {k:.6f} folha^-1")
    print(f"R² = {r2:.4f}")

    plt.figure(figsize=(8, 5))

    # Valores experimentais
    plt.scatter(
        camadas[mascara],
        intensidades[mascara],
        color="blue",
        s=65,
        label="Dados experimentais"
    )

    # Curva exponencial
    plt.plot(
        x_modelo,
        y_modelo,
        color="black",
        linestyle="--",
        label=f"Ajuste exponencial (k={k:.3f})"
    )

    plt.xlabel("Número de folhas")
    plt.ylabel("Intensidade relativa - Canal B")
    plt.title("Ajuste exponencial - Laser azul")

    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RAIZ_PROJETO / "resultados/atenuacao/azul/coeficiente_atenuacao_azul.png",
        dpi=300
    )

    plt.show()
