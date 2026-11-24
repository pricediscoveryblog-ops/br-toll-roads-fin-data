# BR Toll Roads Fin Data

App Streamlit público do Price Discovery para concessões rodoviárias brasileiras (federais/ANTT, SP/ARTESP e MG/Artemig).

Abas: visão geral e mapa (legenda por grupo; consolidado de tráfego, receita e EBITDA ajustados, capex, capex de
manutenção, dívida bruta/líquida e custo da dívida com spread sobre o soberano), operacional por SPE (tráfego vs PIB,
financeiro ajustado vs PIB nominal/IPCA/dólar, prazos e ciclo, capex por veículo equivalente e por km, ICSD, DL/EBITDA
com e sem IFRS 16, estrutura societária e peso no grupo), comparar SPEs (ranking por indicador), Juros (ETTJ) e Macro.

Dados: coletor público `pricediscoveryblog-ops/coletor-concessoes` (branches `dados` e `dados-financeiro`, cadastro na
`main`) e séries de juros/macro do Price Discovery Scanning (`pricediscoveryblog-ops/universo-setorial`, `saidas/`).
`macro_aba.py` é cópia do módulo do Price Discovery Scanning. Métricas calculadas no app são cálculo próprio.

Mapa (traçado das rodovias): `geo/tracados.geojson`, gerado por `scripts/gerar_tracados.py` e atualizado pelo workflow
`.github/workflows/tracados.yml` (semanal, manual ou ao mudar o script). Federais: km inicial/final dos trechos concedidos
da ANTT (dataset `trecho-concedido`, via coletor) cortados sobre a geometria do SNV do DNIT (padrão: SNV 202604A,
espelho público `lucaasmeiraf/snv-geojson`; `SNV_URL` troca a fonte). SP: KMZ por lote do dataset `malha-rodoviaria` da
ARTESP. Contratos federais encerrados e BRs delegadas a MG: critérios em `geo/complementos_tracados.csv` (cálculo próprio).
Ainda sem traçado: rodovias estaduais dos lotes do PR (PR-xxx) e lotes estaduais de MG (MG-050, Sul de Minas, Vias do
Café, Via Liberdade, Via Nascentes) e lotes antigos da ARTESP sem KMZ (TEBE, Triângulo do Sol, ViaOeste).
