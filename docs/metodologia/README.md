# Metodologia e estado das etapas

## Processamento inicial

Scripts exploratórios para leitura de imagem, estatísticas RGB, mapa de pixels e inspeção dos arquivos CMA/BMP. `mapaPixels.py` referencia `fotoCamera6.png`, ausente no conjunto versionado; a execução depende desse arquivo ou de ajuste manual da entrada.

## Caracterização do feixe

`medirDiametro.py` e `medirDiametroComentado.py` leem o CMA em `dados/imagens_originais/` e gravam o CSV em `resultados/caracterizacao_feixe/`.

## Atenuação

Os scripts azul, verde e verde inclinado calculam estatísticas de intensidade e ajustes conforme as implementações existentes. A seleção de ROI é interativa. O script verde inclinado salva resultados na pasta verde. Resultados presentes no Git são históricos e não constituem por si só validação experimental final.

## Análise espacial

O script principal gera mapas e momentos por canal a partir das imagens e seleciona ROIs interativamente. As imagens estão referenciadas em `dados/imagens_originais`; saídas padrão em `resultados/analise_espacial/<cor>/`. `validar_etapa3_npz.py` recalcula P, centroides e deltas sem seleção de ROI e compara com os CSV originais. Essa checagem é computacional e não substitui validação experimental.

## Etapa 5 — coordenadas RGB normalizadas

Os arquivos `I_xy_*.npz` da etapa 4 contêm as coordenadas espaciais e as intensidades digitais `I_R`, `I_G` e `I_B` dentro da ROI. A etapa 4 subtrai o fundo por canal e trunca valores negativos para zero. A etapa 5 usa esses valores diretamente para comparar a participação relativa dos canais.

Para cada pixel, sejam `S = R + G + B` e `L` o limiar exploratório em níveis digitais somados. Para `S > L`, calculam-se `r = R/S`, `g = G/S` e `b = 1 - r - g`. Os demais pixels recebem `NaN` nas três coordenadas. A coordenada global corresponde à média ponderada por `S` dos pixels válidos. Assim, `r_global = ΣR/ΣS`, `g_global = ΣG/ΣS` e `b_global = 1 - r_global - g_global`, com as somas sobre a mesma máscara.

Da raiz do projeto, execute `python -m unittest discover -s src/05_coordenadas_cromaticas -v` para verificar os casos sintéticos e os NPZ dos dois lasers. Os seis testes verificaram os extremos RGB, cinza, soma unitária, faixa, zeros, entradas inválidas e preservação dos arrays. Nas seis imagens de cada laser, as coordenadas globais com limiar zero coincidiram com as proporções calculadas a partir da coluna `P` da etapa 4, com tolerância absoluta de `1e-12`.

O comando `python src/05_coordenadas_cromaticas/analisar_sensibilidade.py` avalia os limiares 0, 5, 10, 20 e 30. O CSV em `resultados/coordenadas_cromaticas/analise_sensibilidade/` registra contagem e porcentagem de pixels válidos na ROI, proporções globais, diferenças assinadas em relação ao limiar zero e fração da soma RGB de referência retida. Os gráficos mostram as três coordenadas em função do número de folhas, com uma curva por limiar e uma figura por laser. Se houver repetições por número de folhas, os gráficos usam a mediana.

Os NPZ normalizados, resumos CSV e gráficos históricos da etapa 5 ficam em `resultados/coordenadas_cromaticas/<cor>/`. O programa `coordenadas_RGB.py --modo ambos --limiar 0` pode regenerá-los em uma cópia de trabalho, pois grava nos mesmos destinos.

A concordância com a etapa 4 valida a implementação numérica. A validação física exige medidas independentes. Sinais fracos são mais sensíveis a ruído e a erros na estimativa do fundo. A saturação comprime a resposta dos canais, e exposição, balanço de branco e sensibilidade espectral da câmera podem alterar as proporções. Trata-se de coordenadas RGB da câmera, sem calibração espectral e sem correspondência direta com coordenadas CIE. Repetições com exposição e balanço de branco fixos, medidas de fundo e inspeção de saturação ajudariam a avaliar a estabilidade experimental. Os limiares exploratórios não estabelecem um limite de detecção.

## Pendências identificadas

- `mapaPixels.py` usa `fotoCamera6.png`, que não existe no repositório.
- As análises interativas não foram executadas automaticamente para evitar selecionar ROIs inadvertidamente.
