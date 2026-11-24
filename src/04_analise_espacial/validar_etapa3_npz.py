"""Etapa 3 da IC: valida momentos dos .npz e compara perfis espaciais.

Use depois de executar analise_espacial_lasers_eixos_na_imagem.py.

Exemplos:
    python validar_etapa3_npz.py --modo azul
    python validar_etapa3_npz.py --modo verde
    python validar_etapa3_npz.py --modo ambos

Resultados em resultados/analise_espacial/<modo>/validacao_etapa3/.
Requer: numpy, pandas, matplotlib.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CANAIS = ("R", "G", "B")
METRICAS = ("P", "x_centroide_px", "y_centroide_px", "delta_x_px", "delta_y_px")


def ler_npz(caminho: Path) -> dict[str, np.ndarray]:
    chaves = ("x_px", "y_px", "I_R", "I_G", "I_B")
    with np.load(caminho, allow_pickle=False) as arquivo:
        ausentes = set(chaves).difference(arquivo.files)
        if ausentes:
            raise ValueError(f"{caminho.name}: faltam {sorted(ausentes)}")
        d = {k: np.asarray(arquivo[k], dtype=np.float64).ravel() for k in chaves}
    tamanhos = {len(v) for v in d.values()}
    if len(tamanhos) != 1 or next(iter(tamanhos)) == 0:
        raise ValueError(f"{caminho.name}: arrays de tamanhos diferentes/vazios")
    if any(not np.isfinite(a).all() for a in d.values()):
        raise ValueError(f"{caminho.name}: valor NaN ou infinito")
    if any(np.any(d[f"I_{c}"] < 0) for c in CANAIS):
        raise ValueError(f"{caminho.name}: intensidade negativa inesperada")
    return d


def calcular(x: np.ndarray, y: np.ndarray, i: np.ndarray) -> dict:
    p = float(i.sum())
    base = {
        "P": p,
        "pixels_zerados_pct": float(100.0 * np.mean(i == 0)),
        "n_pixels": int(i.size),
        "x_centroide_px": np.nan,
        "y_centroide_px": np.nan,
        "delta_x_px": np.nan,
        "delta_y_px": np.nan,
    }
    if p <= 0:
        return base
    mx = float(np.dot(x, i) / p)
    my = float(np.dot(y, i) / p)
    vx = float(np.dot((x - mx) ** 2, i) / p)
    vy = float(np.dot((y - my) ** 2, i) / p)
    base.update(x_centroide_px=mx, y_centroide_px=my,
                delta_x_px=float(np.sqrt(max(vx, 0))),
                delta_y_px=float(np.sqrt(max(vy, 0))))
    return base


def camadas_arquivo(nome: str) -> int:
    match = re.search(r"\d+", nome)
    if match is None:
        raise ValueError(f"Arquivo sem numero de camadas: {nome}")
    return int(match.group())


def gerar_perfil(arquivos: list[tuple[Path, dict]], eixo: str, canal: str,
                 destino: Path, bin_px: float, modo: str) -> None:
    """Densidade espacial marginal: soma de I por bin / (P * largura do bin)."""
    todas_coord = np.concatenate([d[f"{eixo}_px"] for _, d in arquivos])
    minimo = float(todas_coord.min())
    maximo = float(todas_coord.max())
    bordas = np.arange(np.floor(minimo / bin_px) * bin_px,
                      np.ceil(maximo / bin_px) * bin_px + 2 * bin_px,
                      bin_px)
    centros = (bordas[:-1] + bordas[1:]) / 2

    fig, ax = plt.subplots(figsize=(9, 5))
    for caminho, d in sorted(arquivos, key=lambda par: camadas_arquivo(par[0].name)):
        coords = d[f"{eixo}_px"]
        intensidade = d[f"I_{canal}"]
        p = float(intensidade.sum())
        if p <= 0:
            continue
        hist, _ = np.histogram(coords, bins=bordas, weights=intensidade)
        densidade = hist / (p * bin_px)
        ax.plot(centros, densidade, label=f"{camadas_arquivo(caminho.stem)} folha(s)", lw=1.5)
    rotulo = f"{eixo}'" if modo == "verde" else eixo
    ax.set(xlabel=f"Coordenada {rotulo} (pixels)",
           ylabel="Perfil marginal normalizado (1/pixel)",
           title=f"Laser {modo} — canal {canal}: perfil ao longo de {rotulo}")
    ax.grid(alpha=0.25)
    ax.legend(title="Camadas", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(destino, dpi=240)
    plt.close(fig)


def processar(modo: str, pasta_raiz: Path | None, bin_px: float) -> None:
    raiz = pasta_raiz if pasta_raiz else Path(__file__).resolve().parents[2] / "resultados" / "analise_espacial" / modo / f"resultados_espaciais_{modo}_eixos"
    pasta_npz = raiz / "dados_I_xy"
    csv_original = raiz / "momentos_por_imagem_e_canal.csv"
    saida = raiz / "validacao_etapa3"
    arquivos = sorted(pasta_npz.glob("*.npz"), key=lambda p: (camadas_arquivo(p.stem), p.stem))
    if not arquivos:
        print(f"[{modo}] Nenhum arquivo .npz encontrado em: {pasta_npz}")
        return
    saida.mkdir(parents=True, exist_ok=True)

    registros = []
    conjuntos = []
    for arq in arquivos:
        d = ler_npz(arq)
        conjuntos.append((arq, d))
        nome_png = arq.name.replace("I_xy_", "", 1).replace(".npz", ".png")
        for canal in CANAIS:
            reg = calcular(d["x_px"], d["y_px"], d[f"I_{canal}"])
            registros.append({"arquivo_npz": arq.name, "arquivo": nome_png,
                              "camadas": camadas_arquivo(arq.name), "canal": canal, **reg})
    df = pd.DataFrame(registros)
    df.to_csv(saida / "momentos_recalculados_npz.csv", index=False)

    if csv_original.exists():
        original = pd.read_csv(csv_original)
        chaves = ["arquivo", "camadas", "canal"]
        if all(c in original.columns for c in chaves + list(METRICAS)):
            comparacao = df.merge(original[chaves + list(METRICAS) +
                                      (["qualidade"] if "qualidade" in original else [])],
                                  on=chaves, how="left", suffixes=("_npz", "_original"), indicator=True)
            discrepancias = []
            for metrica in METRICAS:
                esperado = pd.to_numeric(comparacao[f"{metrica}_original"], errors="coerce")
                atual = pd.to_numeric(comparacao[f"{metrica}_npz"], errors="coerce")
                diferenca = (atual - esperado).abs()
                comparacao[f"dif_{metrica}"] = diferenca
                # Tolerancia absoluta e relativa para comparar floats.
                ok = np.isclose(atual, esperado, atol=1e-7, rtol=1e-8, equal_nan=True)
                discrepancias.append(~ok)
            comparacao["coincide"] = (comparacao["_merge"] == "both") & ~np.any(discrepancias, axis=0)
            comparacao.to_csv(saida / "comparacao_momentos.csv", index=False)
            n_ok = int(comparacao["coincide"].sum())
            print(f"[{modo}] Conferência NPZ x CSV original: {n_ok}/{len(comparacao)} linhas coincidentes.")
            if n_ok != len(comparacao):
                print(f"[{modo}] Revisar as linhas divergentes em comparacao_momentos.csv")
        else:
            print(f"[{modo}] CSV original não possui todas as colunas para a comparação.")
    else:
        print(f"[{modo}] CSV original ausente: {csv_original}. Momentos NPZ foram calculados.")

    principal = "B" if modo == "azul" else "G"
    for eixo in "xy":
        gerar_perfil(conjuntos, eixo, principal,
                     saida / f"perfil_{eixo}_canal_{principal}.png", bin_px, modo)
    print(f"[{modo}] {len(arquivos)} NPZ lidos. Resultado: {saida.resolve()}")
    print(df[df["canal"] == principal][["camadas", "P", "delta_x_px", "delta_y_px", "pixels_zerados_pct"]].round(3).to_string(index=False))
    print("  Observação: perfis normalizados mostram FORMA, não potência total.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modo", choices=("azul", "verde", "ambos"), default="ambos")
    parser.add_argument("--raiz", type=Path, help="Pasta de resultados (apenas para um modo)")
    parser.add_argument("--bin-px", type=float, default=3.0, help="Largura da faixa do perfil em pixels (padrao 3)")
    args = parser.parse_args()
    if args.bin_px <= 0:
        parser.error("--bin-px deve ser positivo")
    if args.raiz and args.modo == "ambos":
        parser.error("--raiz requer --modo azul ou verde")
    for modo in (("azul", "verde") if args.modo == "ambos" else (args.modo,)):
        processar(modo, args.raiz, args.bin_px)


if __name__ == "__main__":
    main()
