from pathlib import Path
import argparse
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Raiz do repositório pesquisa-IC
RAIZ = Path(__file__).resolve().parents[2]


def numero_camadas(nome):
    encontrado = re.search(r"I_xy_(\d+)", nome)

    if encontrado is None:
        raise ValueError(
            f"Nao foi possivel identificar as camadas: {nome}"
        )

    return int(encontrado.group(1))


def calcular_coordenadas(R, G, B, limiar=0.0):
    """
    Calcula r, g e b para cada pixel.

    Os canais de entrada sao intensidades RGB
    corrigidas, provenientes dos arquivos NPZ.

    Pixels com R + G + B <= limiar sao invalidos.
    """

    R = np.asarray(R, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    if not (R.shape == G.shape == B.shape):
        raise ValueError("Os canais RGB devem ter a mesma forma.")

    soma = R + G + B

    validos = (
        np.isfinite(R)
        & np.isfinite(G)
        & np.isfinite(B)
        & (soma > limiar)
    )

    r = np.full(R.shape, np.nan, dtype=np.float64)
    g = np.full(G.shape, np.nan, dtype=np.float64)
    b = np.full(B.shape, np.nan, dtype=np.float64)

    # Coordenadas cromaticas por pixel
    r[validos] = R[validos] / soma[validos]
    g[validos] = G[validos] / soma[validos]
    b[validos] = 1.0 - r[validos] - g[validos]

    return r, g, b, soma, validos


def processar_laser(modo, limiar=0.0):
    # Pastas de entrada dos NPZ
    pasta_entrada = (
        RAIZ
        / "resultados"
        / "analise_espacial"
        / modo
        / f"resultados_espaciais_{modo}_eixos"
        / "dados_I_xy"
    )

    # Nova pasta de resultados
    pasta_saida = (
        RAIZ
        / "resultados"
        / "coordenadas_cromaticas"
        / modo
    )

    pasta_saida.mkdir(parents=True, exist_ok=True)

    pasta_npz = pasta_saida / "dados_rgb_normalizados"
    pasta_npz.mkdir(exist_ok=True)

    arquivos = sorted(
        pasta_entrada.glob("I_xy_*.npz"),
        key=lambda p: (numero_camadas(p.stem), p.stem)
    )

    if not arquivos:
        print(f"Nenhum arquivo NPZ encontrado: {pasta_entrada}")
        return

    registros = []

    for arquivo in arquivos:
        with np.load(arquivo, allow_pickle=False) as dados:
            x = dados["x_px"]
            y = dados["y_px"]

            R = dados["I_R"]
            G = dados["I_G"]
            B = dados["I_B"]

        # Calcular coordenadas por pixel
        r, g, b, soma, validos = calcular_coordenadas(
            R, G, B, limiar
        )

        n_camadas = numero_camadas(arquivo.stem)

        # Salvar os valores de TODOS os pixels da ROI
        destino = pasta_npz / (
            f"rgb_normalizado_{arquivo.stem}.npz"
        )

        np.savez_compressed(
            destino,
            x_px=x,
            y_px=y,
            r=r,
            g=g,
            b=b,
            soma_rgb=soma,
            validos=validos
        )

        n_total = len(soma)
        n_validos = int(np.sum(validos))

        # Coordenadas globais ponderadas pela intensidade
        # r_global = soma(R) / soma(R + G + B)
        # g_global = soma(G) / soma(R + G + B)

        if n_validos > 0:
            soma_total = np.sum(soma[validos])

            r_global = np.sum(R[validos]) / soma_total
            g_global = np.sum(G[validos]) / soma_total
            b_global = 1.0 - r_global - g_global
        else:
            r_global = np.nan
            g_global = np.nan
            b_global = np.nan

        registros.append({
            "laser": modo,
            "arquivo": arquivo.name,
            "camadas": n_camadas,
            "pixels_total": n_total,
            "pixels_validos": n_validos,
            "pixels_validos_pct": (
                100.0 * n_validos / n_total
                if n_total else np.nan
            ),
            "r_global": r_global,
            "g_global": g_global,
            "b_global": b_global
        })

        print(
            f"{modo} | {n_camadas} folha(s): "
            f"r={r_global:.4f}, "
            f"g={g_global:.4f}, "
            f"b={b_global:.4f} | "
            f"pixels validos={n_validos}/{n_total}"
        )

    # Salvar resumo por imagem
    df = pd.DataFrame(registros)
    df = df.sort_values(["camadas", "arquivo"])

    df.to_csv(
        pasta_saida / "coordenadas_rgb_por_imagem.csv",
        index=False
    )

    # Se houver repeticoes, usar mediana para o grafico
    resumo = (
        df.groupby("camadas", as_index=False)
        [["r_global", "g_global", "b_global"]]
        .median()
    )

    # Grafico das coordenadas cromaticas
    plt.figure(figsize=(9, 6))

    for canal, cor in [
        ("r_global", "red"),
        ("g_global", "green"),
        ("b_global", "blue")
    ]:
        plt.plot(
            resumo["camadas"],
            resumo[canal],
            "o-",
            color=cor,
            label=canal[0]
        )

    plt.xlabel("Numero de folhas")
    plt.ylabel("Coordenada RGB normalizada")
    plt.title(f"Coordenadas cromaticas - laser {modo}")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        pasta_saida / "coordenadas_rgb_por_folha.png",
        dpi=220
    )

    plt.close()

    print(f"\nResultados salvos em: {pasta_saida}\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--modo",
        choices=["azul", "verde", "ambos"],
        default="ambos"
    )

    parser.add_argument(
        "--limiar",
        type=float,
        default=0.0,
        help="Soma RGB minima para considerar um pixel valido"
    )

    args = parser.parse_args()

    if args.limiar < 0:
        parser.error("--limiar deve ser >= 0")

    modos = (
        ["azul", "verde"]
        if args.modo == "ambos"
        else [args.modo]
    )

    for modo in modos:
        processar_laser(modo, args.limiar)


if __name__ == "__main__":
    main()