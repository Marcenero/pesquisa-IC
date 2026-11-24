# Iniciação Científica — caracterização de lasers

Repositório de trabalho semestral para processamento de imagens, caracterização de feixe, análise RGB e estudos de atenuação e distribuição espacial dos lasers azul e verde.

## Organização

- `src/01_processamento_inicial/`: inspeção de pixels, imagens e arquivos CMA/BMP.
- `src/02_caracterizacao_feixe/`: scripts e versões comentadas para estimar o diâmetro do feixe.
- `src/03_atenuacao/`: análises RGB e coeficiente efetivo de atenuação.
- `src/04_analise_espacial/`: momentos, centroides, deltas e validação dos NPZ.
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

## Estado científico

O repositório preserva resultados e versões experimentais. A presença de um resultado versionado não implica validação experimental final. Consulte `docs/metodologia/README.md` para limitações conhecidas. Os algoritmos, ROIs, dados e valores experimentais foram mantidos nesta reorganização.
