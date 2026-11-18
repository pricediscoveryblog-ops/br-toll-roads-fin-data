# BR Toll Roads Fin Data

App Streamlit público do Price Discovery para concessões rodoviárias brasileiras (federais/ANTT, SP/ARTESP e MG/Artemig).

Abas: visão geral e mapa (legenda por grupo; consolidado de tráfego, receita e EBITDA ajustados, capex, capex de
manutenção, dívida bruta/líquida e custo da dívida com spread sobre o soberano), operacional por SPE (tráfego vs PIB,
financeiro ajustado vs PIB nominal/IPCA/dólar, prazos e ciclo, capex por veículo equivalente e por km, ICSD, DL/EBITDA
com e sem IFRS 16, estrutura societária e peso no grupo), comparar SPEs (ranking por indicador), Juros (ETTJ) e Macro.

Dados: coletor público `pricediscoveryblog-ops/coletor-concessoes` (branches `dados` e `dados-financeiro`, cadastro na
`main`) e séries de juros/macro do Price Discovery Scanning (`pricediscoveryblog-ops/universo-setorial`, `saidas/`).
`macro_aba.py` é cópia do módulo do Price Discovery Scanning. Métricas calculadas no app são cálculo próprio.
