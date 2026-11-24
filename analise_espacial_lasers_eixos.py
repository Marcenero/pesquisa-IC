"""Analise espacial I(x,y), potencia integrada, centroides e larguras RGB.

Usos:
    python analise_espacial_lasers.py --modo azul
    python analise_espacial_lasers.py --modo verde

Dependencias: pip install opencv-python numpy pandas matplotlib

- Azul: ROI retangular fixa, escolhida ao redor da mancha circular; sem mascara.
- Verde: elipse fixa ajustada por cv2.fitEllipse na imagem de referencia.
- Os momentos usam a imagem ORIGINAL: o laser verde e alinhado por uma
  ROTACAO DE COORDENADAS, sem interpolar intensidades.
- No verde, x' acompanha o EIXO MAIOR (parte mais longa da elipse) e y' o MENOR.
- Desenha os eixos na foto original para cada numero de folhas.
- I(x,y) = max(valor_RGB(x,y) - media_fundo_RGB, 0).
  Isto e uma intensidade digital corrigida POSITIVA, nao potencia optica em watts.
- Larguras Delta x/y sao desvios padrao espaciais ponderados por I, em pixels.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PASTAS_PADRAO = {
    "azul": Path("imagens/camadasEscuraUV"),
    "verde": Path("imagens/camadasEscuraVerde"),
}
CORES = {"R": "red", "G": "green", "B": "blue"}
MAPAS_CORES = {"R": "Reds", "G": "Greens", "B": "Blues"}
MARGEM_ELIPSE = 1.20  # Mesmo criterio do codigo anterior para o verde


def numero_camadas(nome: str) -> int:
    encontrado = re.search(r"\d+", nome)
    if encontrado is None:
        raise ValueError(f"Nome sem numero de camadas: {nome}")
    return int(encontrado.group())


def roi_manual(preview: np.ndarray, escala: float, titulo: str) -> tuple[int, int, int, int]:
    print(f"Selecione {titulo} e pressione ENTER ou ESPACO.")
    x, y, w, h = cv2.selectROI(titulo, preview, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(titulo)
    if w <= 0 or h <= 0:
        raise RuntimeError(f"ROI nao selecionada: {titulo}")
    return tuple(round(v / escala) for v in (x, y, w, h))


def validar_roi(roi: tuple[int, int, int, int], largura: int, altura: int):
    x, y, w, h = roi
    if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > largura or y + h > altura:
        raise ValueError(f"ROI invalida/fora da imagem: {roi}, imagem {largura}x{altura}")


def mascara_verde(
    imagem_rgb: np.ndarray,
    roi_laser: tuple[int, int, int, int],
    roi_fundo: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple, float]:
    """Reaproveita a logica da elipse do ensaio verde anterior."""
    x, y, w, h = roi_laser
    xb, yb, wb, hb = roi_fundo
    regiao = imagem_rgb[y:y+h, x:x+w].astype(np.float32)
    fundo = imagem_rgb[yb:yb+hb, xb:xb+wb].astype(np.float32)
    r, g = regiao[:, :, 0], regiao[:, :, 1]
    fundo_g = float(fundo[:, :, 1].mean())

    limiar_g = max(20.0, fundo_g + 15.0)
    binaria = ((g > limiar_g) & (g > r + 20.0)).astype(np.uint8) * 255
    binaria = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        raise RuntimeError("Nao foi possivel detectar a mancha verde na ROI selecionada.")
    maior = max(contornos, key=cv2.contourArea)
    if len(maior) < 5 or cv2.contourArea(maior) < 30:
        raise RuntimeError("Contorno verde insuficiente para ajustar uma elipse.")

    (cx, cy), (d1, d2), angulo = cv2.fitEllipse(maior)
    elipse = ((float(cx + x), float(cy + y)),
              (float(d1 * MARGEM_ELIPSE), float(d2 * MARGEM_ELIPSE)),
              float(angulo))
    mascara = np.zeros(imagem_rgb.shape[:2], np.uint8)
    cv2.ellipse(mascara, elipse, 255, thickness=-1)
    if not np.any(mascara):
        raise RuntimeError("Mascara eliptica ficou vazia.")

    # Direcao do MAIOR eixo no referencial da imagem, com y apontando para baixo.
    angulo_maior = (angulo if d1 >= d2 else angulo + 90.0) % 180.0
    return mascara.astype(bool), elipse, float(angulo_maior)


def vetores_eixos_locais(
    modo: str, angulo_maior: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Direcoes unitarias de x' e y', em coordenadas cartesianas (y para cima).

    Azul: x horizontal, y vertical.
    Verde: x' ao longo do MAIOR eixo da elipse, y' do MENOR eixo.
    O par de direcoes e ortogonal e preserva orientacao cartesiana.
    """
    if modo == "azul":
        return np.array([1.0, 0.0]), np.array([0.0, 1.0])

    if angulo_maior is None:
        raise ValueError("O modo verde exige o angulo do eixo maior da elipse.")

    # OpenCV: angulo positivo gira para y para baixo; aqui y e para cima.
    theta = np.deg2rad(angulo_maior)
    eixo_x = np.array([np.cos(theta), -np.sin(theta)], dtype=float)

    # Convencao visual: +x' aponta para a direita; para elipse quase vertical,
    # +x' aponta para cima (evita troca de sentido por arredondamento).
    if (abs(eixo_x[0]) < 0.1 and eixo_x[1] < 0) or (abs(eixo_x[0]) >= 0.1 and eixo_x[0] < 0):
        eixo_x = -eixo_x
    eixo_y = np.array([-eixo_x[1], eixo_x[0]], dtype=float)
    return eixo_x, eixo_y


def coordenadas_alinhadas(
    mascara: np.ndarray, centro: tuple[float, float], modo: str,
    angulo_maior: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coordenadas locais dos pixels ORIGINAIS; centro da ROI como origem.

    No verde, x' e paralelo ao eixo MAIOR; y' ao eixo MENOR.
    A foto nao e rotacionada nem interpolada para os calculos.
    """
    linhas, colunas = np.nonzero(mascara)
    dx = colunas.astype(float) - centro[0]
    dy = centro[1] - linhas.astype(float)  # y cartesiano positivo para cima
    ex, ey = vetores_eixos_locais(modo, angulo_maior)
    x_local = ex[0] * dx + ex[1] * dy
    y_local = ey[0] * dx + ey[1] * dy
    return linhas, colunas, x_local, y_local


def imagem_com_eixos(
    imagem_bgr: np.ndarray,
    centro: tuple[float, float],
    modo: str,
    angulo_maior: float | None,
    roi_laser: tuple[int, int, int, int],
    roi_fundo: tuple[int, int, int, int],
    elipse: tuple | None,
    centroide_local: tuple[float, float] | None = None,
) -> np.ndarray:
    """Desenha a ROI e os eixos x'/y' SOBRE a foto (somente visualizacao).

    Eixos e centroides sao desenhados depois das medidas: a imagem ORIGINAL
    continua sendo usada nos calculos de P, centroides e deltas.
    """
    vis = imagem_bgr.copy()
    altura, largura = vis.shape[:2]
    ex, ey = vetores_eixos_locais(modo, angulo_maior)

    if elipse is not None:
        cv2.ellipse(vis, elipse, (0, 220, 255), 2)  # contorno amarelo
        (_centro, tamanhos, _angulo) = elipse
        maior, menor = max(tamanhos), min(tamanhos)
        comp_x = max(32, 0.67 * maior)
        comp_y = max(26, 0.72 * menor)
    else:
        xx, yy, ww, hh = roi_laser
        cv2.rectangle(vis, (xx, yy), (xx + ww - 1, yy + hh - 1), (0, 220, 255), 2)
        comp_x = max(32, 0.65 * ww)
        comp_y = max(26, 0.65 * hh)

    xb, yb, wb, hb = roi_fundo
    cv2.rectangle(vis, (xb, yb), (xb + wb - 1, yb + hb - 1), (255, 80, 20), 2)

    def ponto(vetor_cart: np.ndarray, passo: float) -> tuple[int, int]:
        # Conversao de y cartesiano (cima) para pixel (baixo).
        return (int(round(centro[0] + passo * vetor_cart[0])),
                int(round(centro[1] - passo * vetor_cart[1])))

    def rotulo(texto: str, p: tuple[int, int], cor: tuple[int, int, int]):
        tx = max(4, min(largura - 45, p[0] + 7))
        ty = max(18, min(altura - 7, p[1] - 7))
        cv2.putText(vis, texto, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(vis, texto, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, cor, 2, cv2.LINE_AA)

    # Eixo x': ciano; eixo y': magenta. Desenhados do lado negativo ate o positivo.
    for vetor, comprimento, cor, texto in [
        (ex, comp_x, (255, 255, 0), "x'" if modo == "verde" else "x"),
        (ey, comp_y, (255, 0, 255), "y'" if modo == "verde" else "y"),
    ]:
        negativo = ponto(vetor, -comprimento)
        positivo = ponto(vetor, comprimento)
        cv2.line(vis, negativo, positivo, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.arrowedLine(vis, negativo, positivo, cor, 2, cv2.LINE_AA, tipLength=0.09)
        rotulo(texto, positivo, cor)
        # Tracos a cada 25 pixels no sistema local, sem alterar as intensidades.
        perpendicular = np.array([-vetor[1], vetor[0]])
        for dist in np.arange(25, comprimento, 25):
            for sinal in (-1, 1):
                meio = ponto(vetor, dist * sinal)
                p1 = (int(round(meio[0] - 4 * perpendicular[0])),
                      int(round(meio[1] + 4 * perpendicular[1])))
                p2 = (int(round(meio[0] + 4 * perpendicular[0])),
                      int(round(meio[1] - 4 * perpendicular[1])))
                cv2.line(vis, p1, p2, cor, 2, cv2.LINE_AA)

    o = (int(round(centro[0])), int(round(centro[1])))
    cv2.circle(vis, o, 6, (0, 0, 0), -1, cv2.LINE_AA)
    cv2.circle(vis, o, 4, (255, 255, 255), -1, cv2.LINE_AA)
    rotulo("0", (o[0] - 23, o[1] + 19), (255, 255, 255))

    if centroide_local is not None and all(np.isfinite(centroide_local)):
        xb_local, yb_local = centroide_local
        cx = centro[0] + xb_local * ex[0] + yb_local * ey[0]
        cy = centro[1] - xb_local * ex[1] - yb_local * ey[1]
        p = (int(round(cx)), int(round(cy)))
        cv2.circle(vis, p, 7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.circle(vis, p, 5, (0, 165, 255), 2, cv2.LINE_AA)
        rotulo("C", (p[0] + 11, p[1] - 10), (0, 165, 255))

    legenda = ("x' = eixo MAIOR | y' = eixo MENOR" if modo == "verde"
               else "x = horizontal | y = vertical")
    # Legenda dimensionada para caber tambem em imagens pequenas.
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    escala_texto = 0.50
    texto_w = cv2.getTextSize(legenda, fonte, escala_texto, 1)[0][0]
    if texto_w > largura - 14:
        escala_texto *= max(0.45, (largura - 14) / texto_w)
        texto_w = cv2.getTextSize(legenda, fonte, escala_texto, 1)[0][0]
    if largura > 50:
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (min(largura - 1, texto_w + 15), 29), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, vis, 0.25, 0, vis)
        cv2.putText(vis, legenda, (7, 19), fonte,
                    escala_texto, (255, 255, 255), 1, cv2.LINE_AA)
    return vis

def calcular_momentos(
    valores_originais: np.ndarray,
    nivel_fundo: float,
    desvio_fundo: float,
    x: np.ndarray, y: np.ndarray,
) -> tuple[dict, np.ndarray]:
    """Calcula os 2 primeiros momentos com intensidades digitais positivas."""
    corrigidos_assinados = valores_originais.astype(np.float64) - nivel_fundo
    pesos = np.maximum(corrigidos_assinados, 0.0)
    potencia = float(np.sum(pesos))  # unidade: nivel digital * pixel^2 (area=1)
    P_assinado = float(np.sum(corrigidos_assinados))
    media_corrigida = float(np.mean(corrigidos_assinados))
    # Apenas um INDICADOR de contraste; NAO e uma SNR formal.
    indicador = media_corrigida / max(desvio_fundo, 1.0)
    qualidade = "sinal baixo: interpretar com cautela" if indicador < 3 else "sinal destacado do fundo"

    if potencia <= 0:
        return ({
            "P": 0.0, "P_assinado": P_assinado,
            "x_centroide_px": np.nan, "y_centroide_px": np.nan,
            "var_x_px2": np.nan, "var_y_px2": np.nan,
            "delta_x_px": np.nan, "delta_y_px": np.nan,
            "cov_xy_px2": np.nan,
            "media_corrigida": media_corrigida,
            "desvio_fundo": desvio_fundo,
            "contraste_sobre_desvio_fundo": indicador,
            "qualidade": "P = 0 (sem sinal positivo)",
        }, pesos)

    x_barra = float(np.sum(x * pesos) / potencia)
    y_barra = float(np.sum(y * pesos) / potencia)
    xv = float(np.sum((x - x_barra)**2 * pesos) / potencia)
    yv = float(np.sum((y - y_barra)**2 * pesos) / potencia)
    cov = float(np.sum((x - x_barra) * (y - y_barra) * pesos) / potencia)
    return ({
        "P": potencia,
        "P_assinado": P_assinado,
        "x_centroide_px": x_barra, "y_centroide_px": y_barra,
        "var_x_px2": xv, "var_y_px2": yv,
        "delta_x_px": float(np.sqrt(max(xv, 0.0))),
        "delta_y_px": float(np.sqrt(max(yv, 0.0))),
        "cov_xy_px2": cov,
        "media_corrigida": media_corrigida,
        "desvio_fundo": desvio_fundo,
        "contraste_sobre_desvio_fundo": indicador,
        "qualidade": qualidade,
    }, pesos)


def mapa_intensidade(
    x: np.ndarray, y: np.ndarray, pesos: np.ndarray, bins: tuple[int, int]
) -> tuple[np.ndarray, list[float]]:
    """Mede intensidade MEDIA por celula do mapa (somente para visualizacao)."""
    xmin, xmax = float(x.min() - 1), float(x.max() + 1)
    ymin, ymax = float(y.min() - 1), float(y.max() + 1)
    intervalo = [[xmin, xmax], [ymin, ymax]]
    soma, _, _ = np.histogram2d(x, y, bins=bins, range=intervalo, weights=pesos)
    contagem, _, _ = np.histogram2d(x, y, bins=bins, range=intervalo)
    mapa = np.full(soma.shape, np.nan)
    np.divide(soma, contagem, out=mapa, where=contagem > 0)
    return mapa.T, [xmin, xmax, ymin, ymax]


def salvar_mapa_rgb(
    pasta: Path, arquivo: Path, n: int,
    x: np.ndarray, y: np.ndarray, pesos_canais: dict[str, np.ndarray],
    momentos_canais: dict[str, dict], modo: str,
):
    largura = max(1, float(np.ptp(x)))
    altura = max(1, float(np.ptp(y)))
    bins = (max(15, min(200, int(np.ceil(largura / 2)))),
            max(15, min(200, int(np.ceil(altura / 2)))))
    fig, eixos = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    for eixo, canal in zip(eixos, "RGB"):
        mapa, extent = mapa_intensidade(x, y, pesos_canais[canal], bins)
        im = eixo.imshow(mapa, origin="lower", extent=extent,
                         cmap=MAPAS_CORES[canal], aspect="equal", interpolation="nearest")
        eixo.scatter([0], [0], marker="+", c="black", s=85, label="Centro ROI")
        xb = momentos_canais[canal]["x_centroide_px"]
        yb = momentos_canais[canal]["y_centroide_px"]
        if np.isfinite(xb) and np.isfinite(yb):
            eixo.scatter([xb], [yb], marker="x", c="orange", s=65,
                         label="Centroide I")
        eixo.set(xlabel="x' (pixels)" if modo == "verde" else "x (pixels)",
                 ylabel="y' (pixels)" if modo == "verde" else "y (pixels)",
                 title=f"Canal {canal}")
        eixo.axhline(0, color="black", lw=.5, alpha=.25)
        eixo.axvline(0, color="black", lw=.5, alpha=.25)
        fig.colorbar(im, ax=eixo, shrink=0.73, label="I corrigida (nivel digital)")
        eixo.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"{modo.capitalize()} | {n} folha(s) | mapa I(x, y) (ROI fixa)")
    fig.savefig(pasta / f"I_xy_{arquivo.stem}.png", dpi=170)
    plt.close(fig)


def salvar_curvas(df: pd.DataFrame, pasta: Path, modo: str):
    # Em caso de repeticoes, agrega mediana por numero de folhas para visualizar.
    serie = df.groupby(["camadas", "canal"], as_index=False).agg(
        delta_x_px=("delta_x_px", "median"),
        delta_y_px=("delta_y_px", "median"),
        P=("P", "median"),
        x_centroide_px=("x_centroide_px", "median"),
        y_centroide_px=("y_centroide_px", "median"),
    )
    eixo_x = "X' (maior)" if modo == "verde" else "X"
    eixo_y = "Y' (menor)" if modo == "verde" else "Y"
    for grandeza, titulo, ylabel, nome in [
        ("delta_x_px", f"Largura espacial no eixo {eixo_x}", "Delta x (pixels)", "delta_x_por_folha.png"),
        ("delta_y_px", f"Largura espacial no eixo {eixo_y}", "Delta y (pixels)", "delta_y_por_folha.png"),
        ("P", "Intensidade integrada da ROI", "P (unidades digitais * pixel²)", "P_por_folha.png"),
        ("x_centroide_px", "Centroide no eixo X", "<x> (pixels)", "centroide_x_por_folha.png"),
        ("y_centroide_px", "Centroide no eixo Y", "<y> (pixels)", "centroide_y_por_folha.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for canal in "RGB":
            linhas = serie[serie["canal"] == canal].sort_values("camadas")
            ax.plot(linhas["camadas"], linhas[grandeza], "o-",
                    color=CORES[canal], label=canal)
        ax.set(xlabel="Numero de folhas", ylabel=ylabel, title=f"{titulo} - Laser {modo}")
        ax.grid(alpha=.3)
        ax.legend(title="Canal")
        fig.tight_layout()
        fig.savefig(pasta / nome, dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=["azul", "verde"], default="azul")
    parser.add_argument("--pasta", type=Path, default=None,
                        help="Pasta das imagens PNG; padrao depende do modo")
    parser.add_argument("--saida", type=Path, default=None,
                        help="Pasta de saida; padrao resultados_espaciais_<modo>_eixos")
    parser.add_argument("--roi-laser", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Opcional: ROI em pixels originais, sem janela interativa")
    parser.add_argument("--roi-fundo", nargs=4, type=int, metavar=("X", "Y", "W", "H"),
                        help="Opcional: ROI fundo em pixels originais")
    parser.add_argument("--sem-janela-validacao", action="store_true")
    args = parser.parse_args()

    pasta = args.pasta or PASTAS_PADRAO[args.modo]
    saida = args.saida or Path(f"resultados_espaciais_{args.modo}_eixos")
    saida.mkdir(parents=True, exist_ok=True)
    pasta_mapas = saida / "mapas_xy"
    pasta_mapas.mkdir(exist_ok=True)
    pasta_numerica = saida / "dados_I_xy"
    pasta_numerica.mkdir(exist_ok=True)
    pasta_fotos_eixos = saida / "fotos_com_eixos"
    pasta_fotos_eixos.mkdir(exist_ok=True)

    arquivos = sorted(pasta.glob("*.png"), key=lambda p: (numero_camadas(p.stem), p.stem))
    if not arquivos:
        raise RuntimeError(f"Nenhuma imagem PNG encontrada em {pasta}")
    img_bgr = cv2.imread(str(arquivos[0]))
    if img_bgr is None:
        raise RuntimeError(f"Nao consegui abrir {arquivos[0]}")
    altura, largura = img_bgr.shape[:2]
    escala = min(1000 / largura, 650 / altura, 1.0)
    preview = cv2.resize(img_bgr, (round(largura*escala), round(altura*escala)),
                         interpolation=cv2.INTER_AREA)

    if args.roi_laser is None:
        roi_laser = roi_manual(preview, escala, "ROI LASER")
    else:
        roi_laser = tuple(args.roi_laser)
    if args.roi_fundo is None:
        roi_fundo = roi_manual(preview, escala, "ROI FUNDO")
    else:
        roi_fundo = tuple(args.roi_fundo)
    validar_roi(roi_laser, largura, altura)
    validar_roi(roi_fundo, largura, altura)
    xb, yb, wb, hb = roi_fundo

    if args.modo == "verde":
        mascara, elipse, angulo_maior = mascara_verde(
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), roi_laser, roi_fundo)
        centro = elipse[0]
    else:
        x, y, w, h = roi_laser
        mascara = np.zeros((altura, largura), dtype=bool)
        mascara[y:y+h, x:x+w] = True  # ROI retangular: SEM mascara circular
        centro = (x+(w-1)/2.0, y+(h-1)/2.0)
        elipse, angulo_maior = None, None

    linhas, colunas, X, Y = coordenadas_alinhadas(mascara, centro, args.modo, angulo_maior)
    print(f"Modo: {args.modo} | ROI pixels: {len(X)} | centro: {centro}")
    if angulo_maior is not None:
        print(f"Eixo maior (angulo na imagem): {angulo_maior:.2f} graus")

    # Eixos desenhados diretamente sobre a imagem de referencia.
    # Isso NAO altera os dados usados na analise.
    vis = imagem_com_eixos(
        img_bgr, centro, args.modo, angulo_maior, roi_laser, roi_fundo, elipse
    )
    cv2.imwrite(str(saida / "validacao_ROI.png"), vis)
    cv2.imwrite(str(saida / "validacao_ROI_com_eixos.png"), vis)
    if not args.sem_janela_validacao:
        cv2.imshow("ROI laser (amarelo) e fundo (azul)",
                   cv2.resize(vis, preview.shape[1::-1]))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    config = {"modo": args.modo, "pasta": str(pasta), "roi_laser": roi_laser,
              "roi_fundo": roi_fundo, "centro_xy_px_original": centro,
              "angulo_eixo_maior_imagem_graus": angulo_maior,
              "mascara": "eliptica" if args.modo == "verde" else "retangular sem mascara",
              "convencao_eixos": "x' no eixo MAIOR, y' no MENOR" if args.modo == "verde" else "x horizontal, y vertical",
              "n_pixels": len(X),
              "modelo_intensidade": "I = max(RGB - media_RGB_fundo, 0) por pixel",
              "unidades_x_y": "pixels", "unidades_P": "nivel_digital*pixel^2"}
    (saida / "configuracao_ROI.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    registros = []
    for arquivo in arquivos:
        img = cv2.imread(str(arquivo))
        if img is None:
            print(f"Aviso: imagem nao pode ser aberta: {arquivo.name}")
            continue
        if img.shape[:2] != (altura, largura):
            raise ValueError(f"Dimensoes de {arquivo.name} diferentes da referencia")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        amostra = rgb[linhas, colunas].astype(float)  # N pixels x 3, imagem original
        fundo = rgb[yb:yb+hb, xb:xb+wb].reshape(-1, 3).astype(float)
        momento_por_canal, peso_por_canal = {}, {}
        for indice, canal in enumerate("RGB"):
            bg = float(np.mean(fundo[:, indice]))
            bg_std = float(np.std(fundo[:, indice]))
            metrica, pesos = calcular_momentos(amostra[:, indice], bg, bg_std, X, Y)
            momento_por_canal[canal] = metrica
            peso_por_canal[canal] = pesos
            registros.append({"fonte": args.modo, "arquivo": arquivo.name,
                              "camadas": numero_camadas(arquivo.name), "canal": canal,
                              "n_pixels_ROI": len(X), "fundo_medio": bg, **metrica})

        # Desenha a geometria local SOBRE A FOTO REAL, para cada N folhas.
        # C mostra o centroide do canal principal (B no azul, G no verde).
        principal = "B" if args.modo == "azul" else "G"
        metrica_principal = momento_por_canal[principal]
        centroide = None
        if (np.isfinite(metrica_principal["x_centroide_px"])
                and np.isfinite(metrica_principal["y_centroide_px"])
                and metrica_principal["contraste_sobre_desvio_fundo"] >= 3):
            centroide = (metrica_principal["x_centroide_px"],
                         metrica_principal["y_centroide_px"])
        foto_eixos = imagem_com_eixos(
            img, centro, args.modo, angulo_maior,
            roi_laser, roi_fundo, elipse, centroide,
        )
        cv2.imwrite(str(pasta_fotos_eixos / f"eixos_{arquivo.stem}.png"), foto_eixos)

        # Arquivo numerico contem TODOS os pixels originais, sem interpolacao
        np.savez_compressed(pasta_numerica / f"I_xy_{arquivo.stem}.npz",
                            x_px=X, y_px=Y, I_R=peso_por_canal["R"],
                            I_G=peso_por_canal["G"], I_B=peso_por_canal["B"])
        salvar_mapa_rgb(pasta_mapas, arquivo, numero_camadas(arquivo.name), X, Y,
                        peso_por_canal, momento_por_canal, args.modo)
        print(f"{arquivo.name}: " + " | ".join(
            f"{c}: P={momento_por_canal[c]['P']:.1f}, "
            f"Dx={momento_por_canal[c]['delta_x_px']:.2f}, "
            f"Dy={momento_por_canal[c]['delta_y_px']:.2f} px"
            for c in "RGB"))

    if not registros:
        raise RuntimeError("Nenhuma imagem foi processada")
    tabela = pd.DataFrame(registros).sort_values(["camadas", "canal", "arquivo"])
    tabela.to_csv(saida / "momentos_por_imagem_e_canal.csv", index=False)
    salvar_curvas(tabela, saida, args.modo)
    print(f"\nResultados salvos em: {saida.resolve()}")
    print(f"Fotos anotadas com os eixos: {(saida / 'fotos_com_eixos').resolve()}")
    print("Confira a coluna 'qualidade': valores fracos de R/G/B podem refletir ruido do fundo.")


if __name__ == "__main__":
    main()