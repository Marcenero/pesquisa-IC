"""Analise da intensidade RGB de um laser verde com ROI eliptica inclinada.



Selecione um retangulo contendo toda a mancha do laser na imagem de referencia,

e depois um retangulo de fundo. A elipse e ajustada somente na referencia e

mantida identica em todas as outras imagens.



Requer: opencv-python, numpy, pandas, matplotlib.

"""



import re

from pathlib import Path



import cv2

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd



RAIZ_PROJETO = Path(__file__).resolve().parents[2]
(RAIZ_PROJETO / "resultados/atenuacao/verde").mkdir(parents=True, exist_ok=True)
PASTA_IMAGENS = RAIZ_PROJETO / "dados/imagens_originais/camadasEscuraVerde"

LARGURA_MAX = 1000

ALTURA_MAX = 650

MARGEM_ELIPSE = 1.20  # 1.20 = 20% maior em cada eixo





def numero_camadas(nome):

    encontrado = re.search(r"\d+", nome)

    if encontrado is None:

        raise ValueError(f"Nao encontrei o numero de camadas em: {nome}")

    return int(encontrado.group())





def estatisticas(canal):

    return {

        "media": float(np.mean(canal)),

        "mediana": float(np.median(canal)),

        "desvio": float(np.std(canal)),

        "maximo": float(np.max(canal)),

        "p99": float(np.percentile(canal, 99)),

        "p999": float(np.percentile(canal, 99.9)),

        "saturados_pct": float(np.mean(canal >= 255) * 100),

    }





def selecionar_roi(titulo, preview, escala):

    print(f"Selecione: {titulo}. Pressione ENTER para confirmar.")

    xp, yp, wp, hp = cv2.selectROI(

        titulo, preview, showCrosshair=True, fromCenter=False

    )

    cv2.destroyWindow(titulo)

    if wp == 0 or hp == 0:

        raise RuntimeError(f"ROI '{titulo}' nao foi selecionada.")

    return (

        round(xp / escala),

        round(yp / escala),

        round(wp / escala),

        round(hp / escala),

    )





def criar_mascara_eliptica(referencia_rgb, retangulo_laser, retangulo_fundo):

    """Identifica a mancha verde e cria uma unica ROI eliptica fixa.



    A segmentacao por cor e usada SOMENTE para achar a geometria na referencia.

    Intensidades sao medidas em TODOS os pixels dentro da elipse, sem aplicar

    limiar por intensidade a cada nova imagem.

    """

    x, y, w, h = retangulo_laser

    xb, yb, wb, hb = retangulo_fundo



    regiao = referencia_rgb[y:y+h, x:x+w].astype(np.float32)

    fundo = referencia_rgb[yb:yb+hb, xb:xb+wb].astype(np.float32)

    if regiao.size == 0 or fundo.size == 0:

        raise RuntimeError("Uma ROI esta fora dos limites da imagem.")



    r = regiao[:, :, 0]

    g = regiao[:, :, 1]

    fundo_g = float(np.mean(fundo[:, :, 1]))



    # Critério para detectar a mancha: verde acima do fundo e do vermelho.

    # Ajuste estes dois incrementos apenas se a deteccao falhar.

    limiar_g = max(20.0, fundo_g + 15.0)

    binaria = np.uint8((g > limiar_g) & (g > r + 20.0)) * 255

    binaria = cv2.morphologyEx(

        binaria, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8)

    )



    contornos, _ = cv2.findContours(

        binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE

    )

    if not contornos:

        raise RuntimeError(

            "Nao detectei a mancha verde. Confira a ROI do laser e os limiares."

        )



    contorno = max(contornos, key=cv2.contourArea)

    if len(contorno) < 5 or cv2.contourArea(contorno) < 30:

        raise RuntimeError("Contorno verde insuficiente para ajustar uma elipse.")



    (cx, cy), (eixo_a, eixo_b), angulo = cv2.fitEllipse(contorno)

    elipse = (

        (float(cx + x), float(cy + y)),

        (float(eixo_a * MARGEM_ELIPSE), float(eixo_b * MARGEM_ELIPSE)),

        float(angulo),

    )



    mascara = np.zeros(referencia_rgb.shape[:2], dtype=np.uint8)

    cv2.ellipse(mascara, elipse, 255, thickness=-1)

    if not np.any(mascara):

        raise RuntimeError("A ROI eliptica ficou vazia.")



    # Angulo aproximado para verticalizar APENAS uma visualizacao, se desejado.

    direcao_eixo_maior = angulo if eixo_a >= eixo_b else angulo + 90.0

    rotacao_vertical = ((direcao_eixo_maior - 90.0 + 90.0) % 180.0) - 90.0

    return mascara > 0, elipse, rotacao_vertical





def rotacionar_para_visualizacao(imagem_bgr, angulo):

    """Rotaciona com canvas expandido, evitando cortar a imagem.



    Nao use os pixels interpolados desta versao para a analise quantitativa.

    """

    altura, largura = imagem_bgr.shape[:2]

    centro = (largura / 2.0, altura / 2.0)

    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)

    c, s = abs(matriz[0, 0]), abs(matriz[0, 1])

    largura_nova = int(np.ceil(altura * s + largura * c))

    altura_nova = int(np.ceil(altura * c + largura * s))

    matriz[0, 2] += largura_nova / 2.0 - centro[0]

    matriz[1, 2] += altura_nova / 2.0 - centro[1]

    return cv2.warpAffine(

        imagem_bgr, matriz, (largura_nova, altura_nova),

        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,

        borderValue=(0, 0, 0),

    )





def main():

    arquivos = sorted(

        PASTA_IMAGENS.glob("*.png"), key=lambda arq: numero_camadas(arq.name)

    )

    if not arquivos:

        raise RuntimeError(f"Nenhuma imagem PNG encontrada em: {PASTA_IMAGENS}")



    referencia_bgr = cv2.imread(str(arquivos[0]))

    if referencia_bgr is None:

        raise RuntimeError(f"Nao foi possivel abrir {arquivos[0]}")

    altura, largura = referencia_bgr.shape[:2]

    escala = min(LARGURA_MAX / largura, ALTURA_MAX / altura, 1.0)

    largura_preview = max(1, round(largura * escala))

    altura_preview = max(1, round(altura * escala))

    preview = cv2.resize(

        referencia_bgr, (largura_preview, altura_preview),

        interpolation=cv2.INTER_AREA,

    )



    # Primeiro retangulo: toda a mancha, mais um pequeno espaco ao redor.

    roi_laser_aproximada = selecionar_roi("ROI LASER", preview, escala)

    roi_fundo = selecionar_roi("ROI FUNDO", preview, escala)



    referencia_rgb = cv2.cvtColor(referencia_bgr, cv2.COLOR_BGR2RGB)

    mascara_laser, elipse, angulo_vertical = criar_mascara_eliptica(

        referencia_rgb, roi_laser_aproximada, roi_fundo

    )

    xb, yb, wb, hb = roi_fundo

    quantidade_pixels = int(np.count_nonzero(mascara_laser))



    print(f"Pixels na ROI eliptica: {quantidade_pixels}")

    print(f"Angulo estimado para verticalizar a imagem: {angulo_vertical:.1f} graus")



    # Inspecao visual da regiao que SERA usada na medicao.

    verificacao = referencia_bgr.copy()

    cv2.ellipse(verificacao, elipse, (0, 255, 255), 2)  # amarelo

    cv2.rectangle(

        verificacao, (xb, yb), (xb + wb, yb + hb), (255, 0, 0), 2

    )  # azul

    cv2.imwrite(str(RAIZ_PROJETO / "resultados/atenuacao/verde/validacao_roi_inclinada.png"), verificacao)

    cv2.imshow(

        "ROI laser (amarelo) | ROI fundo (azul)",

        cv2.resize(verificacao, (largura_preview, altura_preview)),

    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()



    # Opcional: salvar copia somente para visualizacao vertical.

    # A quantificacao continua nas imagens ORIGINAIS.

    vertical = rotacionar_para_visualizacao(referencia_bgr, angulo_vertical)

    cv2.imwrite(str(RAIZ_PROJETO / "resultados/atenuacao/verde/laser_vertical_visualizacao.png"), vertical)



    resultados = []

    for arquivo in arquivos:

        camadas = numero_camadas(arquivo.name)

        imagem_bgr = cv2.imread(str(arquivo))

        if imagem_bgr is None:

            print(f"Aviso: nao foi possivel carregar {arquivo.name}")

            continue

        if imagem_bgr.shape[:2] != (altura, largura):

            raise ValueError(

                f"{arquivo.name} tem dimensoes diferentes. "

                "Alinhe/redimensione as imagens antes de aplicar a ROI fixa."

            )



        imagem = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)

        pixels_laser = imagem[mascara_laser].astype(np.float32)  # N x 3

        pixels_fundo = imagem[yb:yb+hb, xb:xb+wb].reshape(-1, 3).astype(np.float32)



        resultado = {"arquivo": arquivo.name, "camadas": camadas}

        for canal, indice in {"R": 0, "G": 1, "B": 2}.items():

            estat_sinal = estatisticas(pixels_laser[:, indice])

            estat_fundo = estatisticas(pixels_fundo[:, indice])

            resultado[f"{canal}_media"] = estat_sinal["media"]

            resultado[f"{canal}_fundo"] = estat_fundo["media"]

            resultado[f"{canal}_corrigido"] = (

                estat_sinal["media"] - estat_fundo["media"]

            )

            resultado[f"{canal}_mediana"] = estat_sinal["mediana"]

            resultado[f"{canal}_desvio"] = estat_sinal["desvio"]

            resultado[f"{canal}_max"] = estat_sinal["maximo"]

            resultado[f"{canal}_p99"] = estat_sinal["p99"]

            resultado[f"{canal}_p999"] = estat_sinal["p999"]

            resultado[f"{canal}_saturados_pct"] = estat_sinal["saturados_pct"]



        resultados.append(resultado)



    if not resultados:

        raise RuntimeError("Nenhuma imagem foi processada.")

    df = pd.DataFrame(resultados).sort_values("camadas").reset_index(drop=True)

    if not (df["camadas"] == 1).any():

        raise ValueError("Preciso de uma imagem com 1 folha para normalizar os dados.")

    referencia = df.loc[df["camadas"] == 1].iloc[0]

    for canal in ["R", "G", "B"]:

        valor_ref = float(referencia[f"{canal}_corrigido"])

        if abs(valor_ref) < 1e-8:

            print(f"Aviso: referencia do canal {canal} quase zero; relativo = NaN")

            df[f"{canal}_relativo"] = np.nan

        else:

            df[f"{canal}_relativo"] = df[f"{canal}_corrigido"] / valor_ref



    colunas = [

        "camadas", "R_corrigido", "G_corrigido", "B_corrigido",

        "R_relativo", "G_relativo", "B_relativo",

        "R_p999", "G_p999", "B_p999",

        "R_saturados_pct", "G_saturados_pct", "B_saturados_pct",

    ]

    print("\n", df[colunas].to_string(index=False))

    df.to_csv(RAIZ_PROJETO / "resultados/atenuacao/verde/resultados_intensidade_inclinada.csv", index=False)



    # Desvio-padrao espacial na ROI, igual ao seu grafico original.

    # NAO equivale a incerteza da media medida nem a repeticoes experimentais.

    plt.figure(figsize=(8, 5))

    for canal, cor in {"R": "red", "G": "green", "B": "blue"}.items():

        if canal == "G":

            valor_ref = abs(float(referencia["G_corrigido"]))

            dispersao_relativa = (

                df["G_desvio"] / valor_ref

                if valor_ref > 1e-8 else np.full(len(df), np.nan)

            )

            plt.errorbar(

                df["camadas"], df["G_relativo"], yerr=dispersao_relativa,

                marker="o", linestyle="-", color=cor, capsize=4,

                elinewidth=1.5, label="G (desvio espacial)",

            )

        else:

            plt.plot(

                df["camadas"], df[f"{canal}_relativo"], marker="o",

                color=cor, label=canal,

            )



    plt.xlabel("Numero de folhas")

    plt.ylabel("Intensidade relativa")

    plt.title("Decaimento da intensidade com o numero de folhas")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.show()

    # ---------------------------------------------------------
    # Coeficiente efetivo de atenuacao: canal verde (520 nm)
    # Modelo: I_rel(N) = exp[-k_G * (N - 1)]
    # Ajuste log-linear com intercepto fixado em zero
    # ---------------------------------------------------------

    camadas = df["camadas"].to_numpy(dtype=float)
    intensidades = df["G_relativo"].to_numpy(dtype=float)

    # Verificacoes: ln(I_rel) requer intensidade positiva e finita.
    mascara_validos = np.isfinite(intensidades) & (intensidades > 0)
    qtd_descartados = int((~mascara_validos).sum())
    if qtd_descartados:
        print(f"Aviso: {qtd_descartados} ponto(s) G ignorado(s): sinal nao positivo/invalido.")

    x = camadas[mascara_validos] - 1.0
    y_log = np.log(intensidades[mascara_validos])

    if len(x) < 3 or np.sum(x**2) == 0:
        print("Pontos positivos insuficientes para estimar k_G.")
    else:
        # Minimos quadrados no dominio logaritmico com origem fixa
        k_g = -np.sum(x * y_log) / np.sum(x**2)

        # R2 calculado na escala original (intensidades relativas)
        y_real = intensidades[mascara_validos]
        y_estimado = np.exp(-k_g * x)
        ss_res = np.sum((y_real - y_estimado)**2)
        ss_tot = np.sum((y_real - y_real.mean())**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        print("\nAJUSTE EXPONENCIAL - LASER VERDE (520 nm)")
        print(f"k_G = {k_g:.6f} folha^-1")
        print(f"R2 (escala original) = {r2:.4f}")

        if k_g <= 0:
            print("Aviso: k_G nao positivo. O sinal nao apresenta decaimento no ajuste.")

        # Dados exportados para o relatorio
        pd.DataFrame([{
            "fonte": "laser_verde_520nm",
            "canal": "G",
            "k_por_folha": k_g,
            "R2_intensidade": r2,
            "n_pontos_ajuste": len(x),
        }]).to_csv(RAIZ_PROJETO / "resultados/atenuacao/verde/coeficiente_atenuacao_verde.csv", index=False)

        # Curva teorica + dados experimentais
        n_modelo = np.linspace(camadas.min(), camadas.max(), 200)
        intensidade_modelo = np.exp(-k_g * (n_modelo - 1.0))

        plt.figure(figsize=(8, 5))
        plt.scatter(
            camadas[mascara_validos], y_real, color="green", s=65,
            label="Canal G - dados experimentais"
        )
        plt.plot(
            n_modelo, intensidade_modelo, color="black", linestyle="--",
            label=f"Ajuste exponencial (k={k_g:.3f} folha^-1)"
        )
        plt.xlabel("Numero de folhas")
        plt.ylabel("Intensidade relativa - canal G")
        plt.title("Ajuste exponencial - laser verde (520 nm)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RAIZ_PROJETO / "resultados/atenuacao/verde/ajuste_exponencial_verde.png", dpi=300)
        plt.show()


if __name__ == "__main__":
    main()
