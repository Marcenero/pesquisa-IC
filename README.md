# Iniciação Científica — caracterização de lasers

Repositório de trabalho semestral para processamento de imagens, caracterização de feixe, análise RGB e estudos de atenuação e distribuição espacial dos lasers azul e verde.

## Organização

- `src/01_processamento_inicial/`: inspeção de pixels, imagens e arquivos CMA/BMP.
- `src/02_caracterizacao_feixe/`: scripts e versões comentadas para estimar o diâmetro do feixe.
- `src/03_atenuacao/`: análises RGB e coeficiente efetivo de atenuação.
- `src/04_analise_espacial/`: momentos, centroides, deltas e validação dos NPZ.
- `src/05_coordenadas_cromaticas/`: coordenadas RGB normalizadas, testes numéricos e análise de sensibilidade ao limiar.
- `dados/imagens_originais/`: imagens de entrada e arquivos experimentais originais.
- `resultados/`: tabelas e gráficos preservados, separados por etapa e por cor.
- `docs/metodologia/`: notas sobre execução e estado de validação.

## Instalação

Use Python 3.10+ e instale as dependências listadas em `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

## Execução

Execute os scripts a partir de qualquer diretório quando usam caminhos relativos ao próprio script, ou invoque pelo caminho a partir da raiz. As análises de atenuação e espacial com seleção de ROI abrem janelas interativas e exigem seleção manual. Exemplo de validação dos resultados espaciais existentes:

```bash
python src/04_analise_espacial/validar_etapa3_npz.py --modo ambos
```

A validação recalcula momentos a partir dos NPZ e compara com CSVs existentes. Resultados gerados ficam em `resultados/analise_espacial/<cor>/resultados_espaciais_<cor>_eixos/validacao_etapa3/`.

## Etapa 5 — coordenadas RGB normalizadas

A etapa compara as contribuições relativas dos canais dos lasers azul e verde. Para cada pixel com `S = R + G + B` acima do limiar, calcula `r = R/S`, `g = G/S` e `b = 1 - r - g`. As coordenadas globais usam os canais integrados sobre os pixels válidos (por exemplo, `r_global = ΣR/ΣS`), o que dá mais peso aos pixels intensos que a média simples das coordenadas por pixel.

As entradas são os NPZ `dados_I_xy/I_xy_*.npz` da etapa 4, em `resultados/analise_espacial/<cor>/resultados_espaciais_<cor>_eixos/`. O programa original gera NPZ por imagem, um CSV por laser e um gráfico por laser em `resultados/coordenadas_cromaticas/<cor>/`. Execute o primeiro comando somente em uma cópia de trabalho se quiser preservar os arquivos históricos nesses destinos.

```bash
python src/05_coordenadas_cromaticas/coordenadas_RGB.py --modo ambos --limiar 0
python -m unittest discover -s src/05_coordenadas_cromaticas -v
python src/05_coordenadas_cromaticas/analisar_sensibilidade.py
```

Os testes conferem normalização, faixa, entradas inválidas e concordância no limiar zero com os valores integrados da etapa 4. A análise exploratória grava um CSV e dois gráficos em `resultados/coordenadas_cromaticas/analise_sensibilidade/`. Para repetir a geração, use uma cópia de trabalho com essa pasta de saída vazia.

Os canais são intensidades digitais RGB corrigidas pelo fundo, com valores negativos truncados para zero na etapa 4. Ruído, saturação e resposta da câmera afetam as proporções. Esses resultados não são medições espectrais calibradas nem coordenadas CIE. Os limiares avaliados são exploratórios.

## Estado científico

O repositório preserva resultados e versões experimentais. A presença de um resultado versionado não implica validação experimental final. Consulte `docs/metodologia/README.md` para limitações conhecidas. Os algoritmos, ROIs, dados e valores experimentais foram mantidos nesta reorganização.
