# Metodologia e estado das etapas

## Processamento inicial

Scripts exploratórios para leitura de imagem, estatísticas RGB, mapa de pixels e inspeção dos arquivos CMA/BMP. `mapaPixels.py` referencia `fotoCamera6.png`, ausente no conjunto versionado; a execução depende desse arquivo ou de ajuste manual da entrada.

## Caracterização do feixe

`medirDiametro.py` e `medirDiametroComentado.py` estimam perfil e diâmetro a partir de dados CMA. Os dois scripts procuram o CMA na raiz do projeto (os dados estão em `dados/imagens_originais/`), portanto ainda requerem ajuste do caminho antes de executar. Os CSV e gráficos já produzidos estão preservados.

## Atenuação

Os scripts azul, verde e verde inclinado calculam estatísticas de intensidade e ajustes conforme as implementações existentes. A seleção de ROI é interativa. O script verde inclinado salva resultados na pasta verde. Resultados presentes no Git são históricos e não constituem por si só validação experimental final.

## Análise espacial

O script principal gera mapas e momentos por canal a partir das imagens e seleciona ROIs interativamente. As imagens estão referenciadas em `dados/imagens_originais`; saídas padrão em `resultados/analise_espacial/<cor>/`. `validar_etapa3_npz.py` recalcula P, centroides e deltas sem seleção de ROI e compara com os CSV originais. Essa checagem é computacional e não substitui validação experimental.

## Pendências identificadas

- `coresPixels.py`, `leitorImagem.py`, `tamanhoArquivo.py` e ambos os scripts de diâmetro mantêm referências históricas relativas à raiz, que precisam ser adaptadas para `dados/imagens_originais/` para execução após a organização.
- `mapaPixels.py` usa `fotoCamera6.png`, que não existe no repositório.
- As análises interativas não foram executadas automaticamente para evitar selecionar ROIs inadvertidamente.
