"""Compara limiares exploratórios de soma RGB usando os NPZ da etapa 4."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coordenadas_RGB import RAIZ, calcular_coordenadas, coordenadas_globais, numero_camadas

LIMIARES = (0, 5, 10, 20, 30)
CANAIS = ("r", "g", "b")


def analisar_laser(laser):
    pasta = RAIZ / "resultados/analise_espacial" / laser / f"resultados_espaciais_{laser}_eixos"
    arquivos = sorted((pasta / "dados_I_xy").glob("I_xy_*.npz"))
    if not arquivos:
        raise FileNotFoundError(f"Nenhum NPZ em {pasta / 'dados_I_xy'}")

    for arquivo in arquivos:
        with np.load(arquivo, allow_pickle=False) as dados:
            R, G, B = (dados[f"I_{canal}"] for canal in "RGB")
        for limiar in LIMIARES:
            _, _, _, soma, validos = calcular_coordenadas(R, G, B, limiar)
            global_rgb = coordenadas_globais(R, G, soma, validos)
            retida = soma[validos].sum()
            if limiar == 0:
                referencia = global_rgb
                soma_total = retida
            n_validos = int(validos.sum())
            yield {
                "laser": laser,
                "arquivo": arquivo.name,
                "camadas": numero_camadas(arquivo.stem),
                "limiar": limiar,
                "pixels_total": soma.size,
                "pixels_validos": n_validos,
                "pixels_validos_pct": 100 * n_validos / soma.size if soma.size else np.nan,
                **{f"{c}_global": v for c, v in zip(CANAIS, global_rgb)},
                **{f"delta_{c}": v - ref for c, v, ref in zip(CANAIS, global_rgb, referencia)},
                "fracao_rgb_retida": retida / soma_total if soma_total > 0 else np.nan,
            }


def plotar(tabela, laser, destino):
    fig, eixos = plt.subplots(1, 3, figsize=(12, 4), sharex=True, layout="constrained")
    resumo = tabela.groupby(["limiar", "camadas"])[[f"{c}_global" for c in CANAIS]].median()
    for eixo, canal in zip(eixos, CANAIS):
        for limiar in LIMIARES:
            serie = resumo.loc[limiar, f"{canal}_global"]
            eixo.plot(serie.index, serie, "o-", label=str(limiar))
        eixo.set(xlabel="Número de folhas", ylabel=f"{canal} global", xticks=serie.index)
        eixo.grid(alpha=0.3)
    eixos[-1].legend(title="Limiar RGB")
    fig.suptitle(f"Sensibilidade das coordenadas globais — laser {laser}")
    fig.savefig(destino, dpi=200)
    plt.close(fig)


def main():
    tabela = pd.DataFrame(
        registro for laser in ("azul", "verde") for registro in analisar_laser(laser)
    ).sort_values(["laser", "camadas", "arquivo", "limiar"])
    destino = RAIZ / "resultados/coordenadas_cromaticas/analise_sensibilidade"
    destino.mkdir(parents=True, exist_ok=True)
    csv = destino / "sensibilidade_por_imagem.csv"
    graficos = {laser: destino / f"sensibilidade_{laser}.png" for laser in ("azul", "verde")}
    for arquivo in (csv, *graficos.values()):
        if arquivo.exists():
            raise FileExistsError(f"Preservando resultado existente: {arquivo}")
    tabela.to_csv(csv, index=False)
    for laser, arquivo in graficos.items():
        plotar(tabela[tabela["laser"] == laser], laser, arquivo)
    print(f"{len(tabela)} linhas salvas em {csv}")


if __name__ == "__main__":
    main()
