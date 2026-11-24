"""BR Toll Roads Fin Data — Price Discovery.

Fontes (todas públicas):
- Coletor pricediscoveryblog-ops/coletor-concessoes: branch `dados` (tráfego ANTT/ARTESP, praças, dados econômicos
  ANTT), branch `dados-financeiro` (DFP/ITR da CVM das SPEs, provisão de manutenção, debêntures SND/ANBIMA, acionistas
  do FRE) e `cadastro/` na main (cadastro mestre e acionistas por curadoria).
- Price Discovery Scanning (pricediscoveryblog-ops/universo-setorial, saidas/ na main): ETTJ ANBIMA, macro
  (IPCA, PIB, PTAX) e, enquanto o bloco financeiro não roda, a base de debêntures como reserva.
Métricas calculadas aqui são marcadas como cálculo próprio.
"""
import io
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import requests
import streamlit as st


# --- Ajuste visual aplicado a TODOS os gráficos Plotly na hora de desenhar (st.plotly_chart / col.plotly_chart):
# (1) legenda horizontal que ficava abaixo do gráfico passa para logo acima da área do gráfico, liberando a base só para
#     os rótulos do eixo x (trimestres/LTM inclinados não ficam mais tampados);
# (2) títulos longos quebram em linhas (<br>) em vez de serem cortados; a margem superior é calculada para caber
#     título + legenda sem sobreposição.
_LIM_TIT = 50  # caracteres por linha de título (gráficos em meia largura têm ~450-550 px)


def _quebrar_titulo(txt: str, lim: int = _LIM_TIT) -> str:
    partes = []
    for bloco in str(txt).split("<br>"):
        linha, linhas = "", []
        for pal in bloco.split(" "):
            if linha and len(linha) + 1 + len(pal) > lim:
                linhas.append(linha)
                linha = pal
            else:
                linha = (linha + " " + pal) if linha else pal
        linhas.append(linha)
        partes.append("<br>".join(linhas))
    return "<br>".join(partes)


def _linhas_legenda(fig, larg: int = 420) -> int:
    """Estimativa conservadora (largura ~420 px) de quantas linhas a legenda horizontal ocupa. O Plotly alinha os itens
    em colunas de largura igual (a do maior item), então itens por linha = largura ÷ maior item."""
    vistos, itens = set(), []
    for t in fig.data:
        n = str(t.name) if t.name else ""
        if n and t.showlegend is not False and n not in vistos:
            vistos.add(n)
            itens.append(48 + 6.5 * len(n))
    if not itens:
        return 0
    por_linha = max(1, int(larg // max(itens)))
    return -(-len(itens) // por_linha)


def _ajustar_fig(fig):
    try:
        lay = fig.layout
        tit = lay.title.text if lay.title is not None else None
        if tit and len(tit) > _LIM_TIT and "<br>" not in tit:
            tit = _quebrar_titulo(tit)
            fig.update_layout(title_text=tit)
        n_tit = (tit.count("<br>") + 1) if tit else 0
        leg = lay.legend
        mover_leg = leg.orientation == "h" and leg.y is not None and leg.y < 0 and leg.yref != "container"
        if not mover_leg and n_tit <= 1:
            return fig
        n_leg = _linhas_legenda(fig) if mover_leg else 0
        h0 = lay.height or 450
        h = h0 + 21 * n_leg + 22 * max(n_tit - 1, 0)
        t_px = 8 + 22 * n_tit + 21 * n_leg + 8
        upd = dict(height=h, margin=dict(t=max(t_px, lay.margin.t or 0)))
        if n_tit:  # título centrado numa posição fixa em pixels a partir do topo (multilinha não é cortado)
            upd["title"] = dict(yref="container", y=1 - (8 + 11 * n_tit) / h, yanchor="middle", automargin=False)
        if mover_leg:
            upd["legend"] = dict(y=1.0, yanchor="bottom", x=0, xanchor="left", yref="paper")
            upd["margin"]["b"] = max(lay.margin.b or 0, 10)
        fig.update_layout(**upd)
    except Exception:
        pass
    return fig


# Instalação refeita a cada execução (o Streamlit Cloud aplica push sem reiniciar o processo): o wrapper chama sempre
# a versão mais recente de _ajustar_fig e o plotly_chart ORIGINAL do Streamlit, sem empilhar wrappers.
from streamlit.delta_generator import DeltaGenerator as _DG

if getattr(st, "_pd_pc_original", None) is None:
    _pc_atual = _DG.plotly_chart
    if getattr(_pc_atual, "__name__", "") == "_plotly_chart_ajustado":  # versão anterior já instalada no processo
        _pc_atual = _pc_atual.__globals__["_pc_original"]
    st._pd_pc_original = _pc_atual
st._pd_ajustar_fig = _ajustar_fig


def _plotly_chart_ajustado(self, figure_or_data, *args, **kwargs):
    if hasattr(figure_or_data, "update_layout") and hasattr(figure_or_data, "layout"):
        st._pd_ajustar_fig(figure_or_data)
    return st._pd_pc_original(self, figure_or_data, *args, **kwargs)


_DG.plotly_chart = _plotly_chart_ajustado
st.plotly_chart = lambda figure_or_data, *a, **k: _plotly_chart_ajustado(st._main, figure_or_data, *a, **k)
st._pd_graficos_ajustados = True

import macro_aba

st.set_page_config(page_title="BR Toll Roads Fin Data", page_icon="🛣️", layout="wide")
RAW = "https://raw.githubusercontent.com/pricediscoveryblog-ops/"
BASE = os.environ.get("DADOS_BASE", RAW + "coletor-concessoes/dados/")
BASE_FIN = os.environ.get("DADOS_FIN", RAW + "coletor-concessoes/dados-financeiro/")
BASE_CAD = os.environ.get("DADOS_CAD", RAW + "coletor-concessoes/main/cadastro/")
BASE_PDS = os.environ.get("DADOS_PDS", RAW + "universo-setorial/main/saidas/")
REGIOES = {"N": "Norte", "NE": "Nordeste", "CO": "Centro-Oeste", "SE": "Sudeste", "S": "Sul"}
REGIAO_POR_UF = {"AC": "N", "AM": "N", "AP": "N", "PA": "N", "RO": "N", "RR": "N", "TO": "N", "AL": "NE", "BA": "NE",
                 "CE": "NE", "MA": "NE", "PB": "NE", "PE": "NE", "PI": "NE", "RN": "NE", "SE": "NE", "DF": "CO",
                 "GO": "CO", "MS": "CO", "MT": "CO", "ES": "SE", "MG": "SE", "RJ": "SE", "SP": "SE", "PR": "S",
                 "RS": "S", "SC": "S"}
PALETA = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
          "#17becf", "#393b79", "#ad494a", "#637939", "#8c6d31", "#7b4173", "#3182bd"]
SEM_GRUPO = "(não identificado)"
TTL = 6 * 3600


# ================================================================== leitura
def _ler_arquivo(url: str, tipo: str, **kw) -> pd.DataFrame:
    try:
        return pd.read_parquet(url) if tipo == "parquet" else pd.read_csv(url, **kw)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(ttl=TTL, show_spinner="Carregando dados…")
def ler(nome: str) -> pd.DataFrame:
    return _ler_arquivo(BASE + nome + ".parquet", "parquet")


@st.cache_data(ttl=TTL, show_spinner="Carregando dados financeiros…")
def ler_fin(nome: str) -> pd.DataFrame:
    return _ler_arquivo(BASE_FIN + nome + ".parquet", "parquet")


@st.cache_data(ttl=TTL)
def ler_cadastro() -> pd.DataFrame:
    c = _ler_arquivo(BASE_CAD + "concessoes.csv", "csv", sep=";", dtype=str, keep_default_na=False)
    if c.empty:                         # reserva: cópia publicada pelo coletor
        c = ler("cadastro")
    for col in ("cnpj", "lat", "lon"):
        if col not in c:
            c[col] = ""
    c["regiao"] = c["uf_principal"].map(REGIAO_POR_UF).fillna("")
    c["grupo"] = c["grupo"].replace("", SEM_GRUPO)
    return c


@st.cache_data(ttl=TTL)
def ler_acionistas_curadoria() -> pd.DataFrame:
    return _ler_arquivo(BASE_CAD + "acionistas.csv", "csv", sep=";", dtype=str, keep_default_na=False)


@st.cache_data(ttl=TTL, show_spinner=False)
def pasta_pds(sub: str, arquivos: tuple) -> str:
    """Baixa as séries do Price Discovery Scanning (macro/juros) para uma pasta local (o módulo macro_aba lê arquivos)."""
    pasta = Path(tempfile.gettempdir()) / f"pds_{sub}"
    pasta.mkdir(parents=True, exist_ok=True)
    for a in arquivos:
        url = BASE_PDS + f"{sub}/{a}"
        try:
            if url.startswith("http"):
                r = requests.get(url, timeout=60)
                if r.status_code == 200:
                    (pasta / a).write_bytes(r.content)
            elif Path(url).exists():
                (pasta / a).write_bytes(Path(url).read_bytes())
        except Exception:  # noqa: BLE001
            pass
    return str(pasta)


def pasta_macro() -> Path:
    return Path(pasta_pds("macro", tuple(macro_aba.ARQS)))


@st.cache_data(ttl=TTL)
def carregar_juros() -> pd.DataFrame:
    p = Path(pasta_pds("juros", ("parametros.csv",)))
    return pd.read_csv(p / "parametros.csv", parse_dates=["data"]) if (p / "parametros.csv").exists() else pd.DataFrame()


URL_UF = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/brazil-states.geojson"


@st.cache_data(ttl=7 * 24 * 3600, show_spinner=False)
def estados_geojson() -> tuple[dict | None, pd.DataFrame]:
    """Divisas estaduais (GeoJSON público dos estados do Brasil, projeto click_that_hood), simplificadas aqui
    (1 a cada 6 vértices, 3 casas decimais) para o mapa carregar rápido. Retorna o GeoJSON e o ponto do rótulo
    (sigla) de cada estado = média dos vértices do maior polígono."""
    try:
        g = requests.get(URL_UF, timeout=60).json()
    except Exception:  # noqa: BLE001
        return None, pd.DataFrame()

    def anel(r):
        pts = r[::6]
        if len(pts) < 4:
            pts = r
        pts = [[round(x, 3), round(y, 3)] for x, y in pts[:-1]] + [[round(r[0][0], 3), round(r[0][1], 3)]]
        return pts
    feats, rot = [], []
    for f in g.get("features", []):
        geo = f["geometry"]
        polys = [geo["coordinates"]] if geo["type"] == "Polygon" else geo["coordinates"]
        polys = [[anel(r) for r in p] for p in polys]
        maior = max(polys, key=lambda p: len(p[0]))[0]
        sigla = f["properties"].get("sigla", "")
        rot.append({"sigla": sigla, "lon": float(np.mean([x for x, _ in maior])), "lat": float(np.mean([y for _, y in maior]))})
        feats.append({"type": "Feature", "properties": {"sigla": sigla},
                      "geometry": {"type": "MultiPolygon", "coordinates": polys}})
    return {"type": "FeatureCollection", "features": feats}, pd.DataFrame(rot)


ARQ_TRACADOS = Path(__file__).resolve().parent / "geo" / "tracados.geojson"
# Ressalvas de traçado por SPE: aparecem no tooltip da linha e numa nota abaixo do mapa quando a SPE está no filtro
NOTA_TRACADO = {
    "M-VLIB": "Via Liberdade: BR-356 pelos km do Ministério dos Transportes; MG-262 (km 0–72,8) e MG-329 (km 100,5–134,5) "
              "atribuídas pelos nomes de local da base do DER-MG — cálculo próprio, a validar no contrato.",
    "M-TRI": "EPR Triângulo: só as BRs delegadas (BR-452 e BR-365); as estaduais do lote (MG-427, MG-190, CMG-452, "
             "CMG-462, LMG-782/798/812) ainda não têm traçado — faltam os km de cada trecho no contrato.",
}


@st.cache_data(ttl=TTL, show_spinner=False)
def tracados_oficiais(versao: tuple = ()) -> tuple[pd.DataFrame, str]:
    """Traçado de cada concessão (geo/tracados.geojson, gerado por scripts/gerar_tracados.py): federais = km dos
    trechos concedidos da ANTT cortados sobre a geometria do SNV/DNIT; SP = KMZ por lote da ARTESP; contratos
    encerrados e BRs delegadas a MG pelos critérios de geo/complementos_tracados.csv. Uma linha por polilinha."""
    try:
        g = json.loads(ARQ_TRACADOS.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return pd.DataFrame(), ""
    linhas = []
    for f in g.get("features", []):
        p = f["properties"]
        if p.get("km_ini") is not None:
            trecho = f"{p['rodovia']}/{p['uf']} km {p['km_ini']:,.1f}–{p['km_fim']:,.1f}".replace(",", "X").replace(".", ",").replace("X", ".")
        else:
            trecho = f"{p['rodovia']}/{p['uf']}"
        for c in f["geometry"]["coordinates"]:
            linhas.append({"id_concessao": p["id_concessao"], "path": c, "trecho": trecho, "fonte": p.get("fonte", "")})
    return pd.DataFrame(linhas), str(g.get("snv", ""))


@st.cache_data(ttl=TTL)
def trafego_mensal() -> pd.DataFrame:
    """Base única praça x mês: veículos equivalentes pagantes, total de veículos e pesados."""
    partes = []
    e = ler("trafego_equivalente")
    if not e.empty:
        e = e[e.id_concessao != ""].copy()
        tipo = e.tipo_veiculo.str.upper()
        e["veic_pesado"] = np.where(tipo.str.startswith("COMERCIAL"), e.volume, 0.0)
        g = e.groupby(["id_concessao", "praca", "mes"], as_index=False).agg(
            veic_eq=("veic_eq", "sum"), veic_total=("volume", "sum"), veic_pesado=("veic_pesado", "sum"))
        g["fonte"] = "ANTT"
        partes.append(g)
    a = ler("trafego_artesp")
    if not a.empty:
        a = a[(a.id_concessao != "") & a.pagante].copy()
        a["veic_total"] = a.veic_leve + a.veic_pesado + a.veic_moto
        g = a.groupby(["id_concessao", "praca", "mes"], as_index=False).agg(
            veic_eq=("veic_eq", "sum"), veic_total=("veic_total", "sum"), veic_pesado=("veic_pesado", "sum"))
        g["fonte"] = "ARTESP"
        partes.append(g)
    t = pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()
    if not t.empty:
        t["mes"] = pd.to_datetime(t["mes"])
    return t


@st.cache_data(ttl=TTL, show_spinner=False)
def ltm_por_spe() -> pd.DataFrame:
    """Veíc. equivalentes em 12 meses acumulados: linhas = mês (calendário completo), colunas = SPE. Mês sem dado
    publicado fica vazio e derruba as somas de 12 meses que o contêm (não completa lacuna)."""
    t = trafego_mensal()
    if t.empty:
        return pd.DataFrame()
    p = t.groupby(["mes", "id_concessao"]).veic_eq.sum().unstack()
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="MS"))
    return p.rolling(12, min_periods=12).sum()


def pracas_mensal(t: pd.DataFrame) -> pd.DataFrame:
    """Veíc. eq. por praça (colunas) e mês (calendário completo). Em mês com dado da SPE, praça sem número = 0 (ainda
    não existia ou foi desativada/renomeada); mês sem nenhum dado da SPE fica vazio."""
    p = t.groupby(["mes", "praca"]).veic_eq.sum().unstack()
    p = p.reindex(pd.date_range(p.index.min(), p.index.max(), freq="MS"))
    tem = p.notna().any(axis=1)
    p.loc[tem] = p.loc[tem].fillna(0.0)
    return p


def agrupar_pracas(p: pd.DataFrame, ordem: pd.Series, n: int) -> pd.DataFrame:
    """Mantém as n praças com maior valor em `ordem` e soma o resto em 'Outras praças'."""
    top = list(ordem.sort_values(ascending=False).index[:n])
    resto = [c_ for c_ in p.columns if c_ not in top]
    out = p[top].copy()
    if resto:
        out["Outras praças"] = p[resto].sum(axis=1, min_count=1)
    return out


def nz(s):
    return s.fillna(0.0)


def div(a, b):
    a, b = pd.to_numeric(a, errors="coerce"), pd.to_numeric(b, errors="coerce")
    return (a / b.where(b > 0)).replace([np.inf, -np.inf], np.nan)


# ================================================================== financeiro (cálculo próprio)
@st.cache_data(ttl=TTL, show_spinner="Calculando métricas financeiras…")
def financeiro(cad: pd.DataFrame, traf: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por SPE (id do cadastro) x período, com as métricas ajustadas de concessão.
    Receita líquida ajustada = receita líquida sem a receita de construção (feito no coletor, ICPC 01).
    EBITDA ajustado = EBITDA (EBIT + D&A, sem margem de construção) + constituição da provisão de manutenção.
    EBITDA ajustado pós-arrendamento = EBITDA ajustado − pagamentos de arrendamento (principal + juros, DFC)."""
    f = ler_fin("financeiro_spe")
    if f.empty:
        return f
    f = f.assign(id=f["ids"].astype(str).str.split("|")).explode("id")
    f = f[f["id"].isin(cad["id"])].copy()
    f["dt_fim"] = pd.to_datetime(f["dt_fim"])
    for c in ("prov_manut_constituicao", "prov_manut_avp", "prov_manut_pago_dfc", "prov_manut_saldo", "capex_manut",
              "divida_cp", "divida_lp", "pagamento_arrendamento", "ir_csll_pagos", "servico_divida_pago", "juros_pagos",
              "capex_organico", "capex_total", "clientes", "fornecedores", "cpv", "fco", "fco_reportado"):
        if c not in f:
            f[c] = np.nan
    if "capex_manut_aprox" not in f:
        f["capex_manut_aprox"] = False
    f["receita_aj"] = f["receita"]
    f["ebitda_aj"] = f["ebitda"] + nz(f["prov_manut_constituicao"])
    f["ebitda_aj_pos_arrend"] = f["ebitda_aj"] - nz(f["pagamento_arrendamento"])
    f["margem_ebitda_aj"] = div(f["ebitda_aj"], f["receita_aj"])
    f["margem_ebitda_aj_pos_arrend"] = div(f["ebitda_aj_pos_arrend"], f["receita_aj"])
    f["capex"] = f["capex_organico"]
    # FCO comparável (Price Discovery Scanning: sem juros e dividendos pagos/recebidos classificados no FCO) menos os
    # pagamentos de arrendamento (principal + juros), para comparar com o EBITDA ajustado pós-arrendamento
    f["fco_comp_pos_arrend"] = f["fco"] - nz(f["pagamento_arrendamento"])
    f["conversao_fco"] = div(f["fco_comp_pos_arrend"], f["ebitda_aj_pos_arrend"])
    # ICSD em duas fórmulas (cálculo próprio):
    #  (1) regra geral: (EBITDA aj. pós-arrend. − IR/CS pagos − capex manut.) ÷ (juros + principal da dívida, DFC)
    #  (2) arrendamento no denominador: (EBITDA aj. − IR/CS pagos − capex manut.) ÷ (juros + principal da dívida +
    #      todos os pagamentos de arrendamento). Quando a DFC só traz uma linha que mistura dívida e arrendamento,
    #      só a fórmula (2) é calculável, com a linha mista no lugar de dívida + arrendamento.
    if "servico_div_arrend_misto" not in f:
        f["servico_div_arrend_misto"] = np.nan
    serv_ok = f["servico_divida_pago"].fillna(0) > 0
    f["icsd_base_mista"] = ~serv_ok & (f["servico_div_arrend_misto"].fillna(0) > 0)
    f["servico_icsd"] = f["servico_divida_pago"]
    f["servico_icsd_arr"] = (f["servico_divida_pago"] + nz(f["pagamento_arrendamento"])).where(
        ~f["icsd_base_mista"], f["servico_div_arrend_misto"] + nz(f["pagamento_arrendamento"]))
    f["cfads"] = f["ebitda_aj_pos_arrend"] - nz(f["ir_csll_pagos"]) - nz(f["capex_manut"])
    f["cfads_sem_manut"] = f["ebitda_aj_pos_arrend"] - nz(f["ir_csll_pagos"])
    f["cfads_arr"] = f["ebitda_aj"] - nz(f["ir_csll_pagos"]) - nz(f["capex_manut"])
    f["cfads_arr_sem_manut"] = f["ebitda_aj"] - nz(f["ir_csll_pagos"])
    f["icsd"] = div(f["cfads"], f["servico_icsd"])
    f["icsd_sem_manut"] = div(f["cfads_sem_manut"], f["servico_icsd"])
    f["icsd_arr"] = div(f["cfads_arr"], f["servico_icsd_arr"])
    f["icsd_arr_sem_manut"] = div(f["cfads_arr_sem_manut"], f["servico_icsd_arr"])
    f["divida_bruta"] = f["divida_bruta_ex_arrend"]
    f["divida_liquida"] = f["div_liq_sem_arrend"]
    f["divida_liquida_ifrs16"] = f["div_liq_com_arrend"]
    fluxo12 = f["tipo"].isin(["anual", "ltm"])
    f["dl_ebitda_aj_ifrs16"] = div(f["divida_liquida_ifrs16"], f["ebitda_aj"]).where(fluxo12)
    f["dl_ebitda_aj_pre_ifrs16"] = div(f["divida_liquida"], f["ebitda_aj_pos_arrend"]).where(fluxo12)
    dias = np.where(f["tipo"] == "trimestral", 91.25, 365.0)
    f["pmr_dias"] = div(f["clientes"], f["receita_aj"]) * dias
    f["pmp_dias"] = div(f["fornecedores"], f["cpv"]) * dias
    f["ciclo_financeiro_dias"] = f["pmr_dias"] - f["pmp_dias"]
    f["pct_divida_cp"] = div(f["divida_cp"], f["divida_cp"].fillna(0) + f["divida_lp"].fillna(0))
    # custo implícito da dívida: juros pagos em 12 meses ÷ dívida bruta média (fim e um ano antes)
    f = f.sort_values(["id", "tipo", "dt_fim"])
    ant = f[["id", "dt_fim", "divida_bruta"]].drop_duplicates(["id", "dt_fim"]).copy()
    ant["dt_fim"] = ant["dt_fim"] + pd.DateOffset(years=1)
    f = pd.merge_asof(f.sort_values("dt_fim"), ant.rename(columns={"divida_bruta": "divida_bruta_ano_ant"}).sort_values("dt_fim"),
                      on="dt_fim", by="id", tolerance=pd.Timedelta(days=20), direction="nearest")
    f["custo_implicito"] = div(f["juros_pagos"], (f["divida_bruta"] + f["divida_bruta_ano_ant"].fillna(f["divida_bruta"])) / 2).where(fluxo12)
    # tráfego do mesmo período (12 meses até a data de fim; trimestre: 3 meses)
    if not traf.empty:
        mens = traf.groupby(["id_concessao", "mes"]).veic_eq.sum()
        veq = []
        for i, d, t in zip(f["id"], f["dt_fim"], f["tipo"]):
            if i not in mens.index.get_level_values(0):
                veq.append(np.nan)
                continue
            s = mens.loc[i]
            n = 3 if t == "trimestral" else 12
            fim = pd.Timestamp(d.year, d.month, 1)
            jan = s[(s.index > fim - pd.DateOffset(months=n)) & (s.index <= fim)]
            veq.append(jan.sum() if len(jan) == n else np.nan)
        f["veic_eq"] = veq
    else:
        f["veic_eq"] = np.nan
    km = f["id"].map(pd.to_numeric(cad.set_index("id")["km_cadastro"], errors="coerce"))
    f["km"] = km
    f["tarifa_media"] = div(f["receita_aj"] * 1e6, f["veic_eq"])
    f["capex_por_veq"] = div(f["capex"] * 1e6, f["veic_eq"])
    f["capex_manut_por_veq"] = div(f["capex_manut"] * 1e6, f["veic_eq"])
    f["capex_por_km"] = div(f["capex"], km)
    f["capex_manut_por_km"] = div(f["capex_manut"], km)
    f["ebitda_aj_por_km"] = div(f["ebitda_aj"], km)
    info = cad.set_index("id")
    for c in ("nome", "grupo", "regiao", "uf_principal", "agencia"):
        f[c] = f["id"].map(info[c])
    f["regiao_nome"] = f["regiao"].map(REGIOES)
    return f.sort_values(["id", "tipo", "dt_fim"]).reset_index(drop=True)


# ================================================================== juros (mesmo modelo do Price Discovery Scanning)
def svensson(t, b1, b2, b3, b4, l1, l2):
    """Taxa (fração a.a., base 252) da ETTJ ANBIMA no prazo t (anos = dias úteis ÷ 252)."""
    t = np.asarray(t, dtype=float)
    a = (1 - np.exp(-l1 * t)) / (l1 * t)
    c = (1 - np.exp(-l2 * t)) / (l2 * t)
    return b1 + b2 * a + b3 * (a - np.exp(-l1 * t)) + b4 * (c - np.exp(-l2 * t))


def curva_juros(pj: pd.DataFrame, d, tipo_c: str, anos) -> np.ndarray:
    linha = {c: r for c, r in pj[pj["data"] == d].set_index("curva").iterrows()}

    def taxa(nome):
        if nome not in linha:
            return np.full(len(np.atleast_1d(anos)), np.nan)
        r = linha[nome]
        return svensson(anos, r["b1"], r["b2"], r["b3"], r["b4"], r["l1"], r["l2"])
    if tipo_c == "pre":
        return taxa("PREFIXADOS")
    if tipo_c == "ipca":
        return taxa("IPCA")
    return (1 + taxa("PREFIXADOS")) / (1 + taxa("IPCA")) - 1


def classe_indexador(ind) -> str:
    u = str(ind or "").strip().upper()
    return "CDI" if u.startswith("DI") else "IPCA" if u.startswith("IPCA") else "PRÉ" if u.startswith("PR") else "Outros"


def pct_di_para_spread(pct, cdi):
    d = (1 + cdi) ** (1 / 252) - 1
    return (1 + d * pct / 100) ** 252 / (1 + cdi) - 1


def rotulo_remuneracao(r) -> str:
    ind, pct, tx = str(r.get("indice") or "").strip(), r.get("percentual_indice"), r.get("taxa_emissao")
    if ind.upper().startswith("DI") and pd.notna(pct) and pct not in (0, 100) and (pd.isna(tx) or tx == 0):
        return f"{pct:g}% DI"
    if pd.notna(tx) and tx != 0:
        return f"{ind} + {tx:g}%".replace(".", ",")
    return ind or "-"


def custo_debentures(dd: pd.DataFrame, pj: pd.DataFrame) -> pd.DataFrame:
    """Custo contratado (emissão) e de mercado (taxa indicativa ANBIMA) de cada debênture e o soberano no mesmo
    prazo (ETTJ ANBIMA). Mesma convenção do Price Discovery Scanning (CDI: spread sobre o CDI; IPCA: taxa real;
    PRÉ: nominal). nominal_* = custo nominal equivalente com a curva de hoje."""
    d = dd.copy()
    d["vencimento"] = pd.to_datetime(d["vencimento"], errors="coerce")
    d["classe"] = d["indice"].map(classe_indexador)
    hoje = pd.Timestamp.today().normalize()
    venc_anos = ((d["vencimento"] - hoje).dt.days / 365.25).clip(lower=1 / 12)
    dur = pd.to_numeric(d.get("duration_anos"), errors="coerce")
    tem_dur = dur.notna() & (dur > 0)
    d["prazo_anos"] = dur.where(tem_dur, venc_anos)
    pre = real = np.full(len(d), np.nan)
    if not pj.empty and len(d):
        datas = sorted(pj["data"].unique())
        ref = pd.to_datetime(d.get("data_anbima"), errors="coerce").max() if "data_anbima" in d else pd.NaT
        cand = [x for x in datas if pd.isna(ref) or pd.Timestamp(x) <= ref]
        dt_c = pd.Timestamp(cand[-1] if cand else datas[-1])
        t = d["prazo_anos"].to_numpy(float)
        pre, real = curva_juros(pj, dt_c, "pre", t), curva_juros(pj, dt_c, "ipca", t)
    d["curva_pre"], d["curva_real"] = pre * 100, real * 100
    impl = pd.Series((1 + pre) / (1 + real) - 1, index=d.index)
    cl = d["classe"]
    pct = pd.to_numeric(d["percentual_indice"], errors="coerce")
    tx = pd.to_numeric(d["taxa_emissao"], errors="coerce")
    mkt = pd.to_numeric(d.get("taxa_indicativa"), errors="coerce")
    ic = d["indice_correcao"].fillna("").astype(str) if "indice_correcao" in d else pd.Series("", index=d.index)
    cdi = pd.Series(pre, index=d.index)
    eh_cdi = cl == "CDI"
    eh_pct = eh_cdi & pct.notna() & ~pct.isin([0, 100]) & (tx.isna() | (tx == 0))
    mkt_pct = eh_cdi & ic.str.contains(r"%\s*(?:do\s+)?DI", case=False, regex=True)
    contr = tx.where(~cl.isin(["IPCA", "PRÉ"]) | (tx != 0))
    contr = contr.where(~eh_pct, pct_di_para_spread(pct, cdi) * 100)
    merc = mkt.where(~mkt_pct, pct_di_para_spread(mkt, cdi) * 100)
    d["taxa_contratada"] = contr.where(cl != "Outros")
    d["taxa_mercado"] = merc.where(cl != "Outros")
    sob = pd.Series(np.nan, index=d.index).mask(cl == "IPCA", d["curva_real"]).mask(cl == "PRÉ", d["curva_pre"])
    d["soberano"] = sob
    d["spread_mercado"] = (d["taxa_mercado"] - sob).where(~eh_cdi, d["taxa_mercado"])
    d["spread_emissao"] = (d["taxa_contratada"] - sob).where(~eh_cdi, d["taxa_contratada"])
    d["taxa_suspeita"] = cl.isin(["IPCA", "PRÉ"]) & ((d["spread_emissao"] < -5) | (d["taxa_contratada"] < 0))

    def nominal(taxa):
        s = pd.Series(np.nan, index=d.index)
        s = s.mask(eh_cdi, ((1 + cdi) * (1 + taxa / 100) - 1) * 100)
        s = s.mask(cl == "IPCA", ((1 + taxa / 100) * (1 + impl) - 1) * 100)
        return s.mask(cl == "PRÉ", taxa)
    d["nominal_emissao"] = nominal(d["taxa_contratada"]).where(~d["taxa_suspeita"])
    d["nominal_mercado"] = nominal(d["taxa_mercado"])
    d["pre_no_prazo"] = d["curva_pre"]
    d["spread_nominal_emissao"] = d["nominal_emissao"] - d["pre_no_prazo"]
    d["spread_nominal_mercado"] = d["nominal_mercado"] - d["pre_no_prazo"]
    d["remuneracao"] = d.apply(rotulo_remuneracao, axis=1)
    return d


def media_pond(v, w):
    v, w = pd.to_numeric(v, errors="coerce"), pd.to_numeric(w, errors="coerce")
    ok = v.notna() & w.notna() & (w > 0)
    return float((v[ok] * w[ok]).sum() / w[ok].sum()) if ok.any() else np.nan


def resumo_custo(d: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    rot = {"CDI": "CDI+", "IPCA": "IPCA+", "PRÉ": "Prefixada", "Outros": "Outros"}
    for c in ["CDI", "IPCA", "PRÉ", "Outros", "Total"]:
        g = d if c == "Total" else d[d["classe"] == c]
        if g.empty:
            continue
        ok, m = g[~g["taxa_suspeita"]], g[g["taxa_mercado"].notna()]
        linhas.append({"Indexador": "Total (nominal equivalente)" if c == "Total" else rot[c], "Qtde": len(g),
                       "Saldo (R$ mm)": g["saldo_rs_mm"].sum(), "Prazo médio (anos)": media_pond(g["prazo_anos"], g["saldo_rs_mm"]),
                       "Emissão (% a.a., conv. do índice)": np.nan if c == "Total" else media_pond(ok["taxa_contratada"], ok["saldo_rs_mm"]),
                       "Mercado (% a.a., conv. do índice)": np.nan if c == "Total" else media_pond(m["taxa_mercado"], m["saldo_rs_mm"]),
                       "Spread emissão s/ soberano (p.p.)": np.nan if c == "Total" else media_pond(ok["spread_emissao"], ok["saldo_rs_mm"]),
                       "Spread mercado s/ soberano (p.p.)": np.nan if c == "Total" else media_pond(m["spread_mercado"], m["saldo_rs_mm"]),
                       "Custo nominal emissão (% a.a.)": media_pond(ok["nominal_emissao"], ok["saldo_rs_mm"]),
                       "Custo nominal mercado (% a.a.)": media_pond(m["nominal_mercado"], m["saldo_rs_mm"]),
                       "Spread nominal emissão s/ pré (p.p.)": media_pond(ok["spread_nominal_emissao"], ok["saldo_rs_mm"]),
                       "Spread nominal mercado s/ pré (p.p.)": media_pond(m["spread_nominal_mercado"], m["saldo_rs_mm"]),
                       "% do saldo com preço ANBIMA": m["saldo_rs_mm"].sum() / g["saldo_rs_mm"].sum() if g["saldo_rs_mm"].sum() else np.nan})
    return pd.DataFrame(linhas)


@st.cache_data(ttl=TTL, show_spinner=False)
def debentures(cad: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Debêntures das SPEs: coletor (SND + ANBIMA, todas as SPEs com CNPJ) ou, na falta, a base do Price Discovery
    Scanning (só SPEs de grupos listados)."""
    d, fonte = ler_fin("debentures_spe"), "coletor-concessoes (SND + ANBIMA)"
    spe = ler_fin("spe_cvm")
    mapa = {}
    for tab in (spe, cad):
        if not tab.empty and "cnpj" in tab:
            for i, c in zip(tab["id"], tab["cnpj"].astype(str)):
                if c and c != "nan":
                    mapa.setdefault(c.zfill(14), set()).add(i)
    if d.empty:
        d = _ler_arquivo(BASE_PDS + "dividas/debentures.csv.gz", "csv", dtype={"cnpj_emissor": str, "cnpj_grupo": str})
        fonte = "Price Discovery Scanning (SND + ANBIMA; só SPEs de grupos listados)"
        if d.empty:
            return d, ""
        d["cnpj_emissor"] = d["cnpj_emissor"].astype(str).str.zfill(14)
        d = d[d["cnpj_emissor"].isin(mapa)].copy()
        d["ids"] = d["cnpj_emissor"].map(lambda c: "|".join(sorted(mapa[c])))
    d = d.assign(id=d["ids"].astype(str).str.split("|")).explode("id")
    d = d[d["id"].isin(cad["id"])]
    return d, fonte


def motivo_sem_icsd(r) -> tuple[str, str]:
    """(motivo, detalhe do período) quando o ICSD fica vazio — diagnóstico próprio sobre os campos DFC/BP da CVM."""
    if pd.isna(r.get("ebitda")):
        return "EBITDA não calculado (DRE/DFC do período incompleta)", ""
    div_b, serv, misto = r.get("divida_bruta"), r.get("servico_divida_pago"), r.get("servico_div_arrend_misto")
    if pd.isna(serv) or serv == 0:
        if pd.isna(div_b) or div_b <= 0:
            return "SPE sem dívida no balanço, não há serviço da dívida a cobrir", ""
        if pd.notna(misto) and misto != 0:
            return ("a DFC paga dívida e arrendamento numa única linha, então a regra geral (arrendamento fora do "
                    "numerador) não é calculável; só a fórmula com arrendamento no denominador aparece"), f"linha mista {fmt_mm(misto)}, dívida {fmt_mm(div_b)}"
        return ("juros e principal pagos não aparecem na DFC: compatível com carência e juros capitalizados, ou linha "
                "com nome não reconhecido pelo coletor (conferir a DFC na CVM)"), f"dívida {fmt_mm(div_b)}"
    if serv < 0:
        return "serviço da dívida com sinal inesperado na DFC", ""
    return "dado insuficiente no período", ""


# ================================================================== estrutura societária (esquema)
def _n(s_) -> str:
    import unicodedata
    return unicodedata.normalize("NFKD", str(s_)).encode("ascii", "ignore").decode().lower()


def _rot(t: str, largura: int = 28) -> str:
    import textwrap
    return "\\n".join(textwrap.wrap(str(t).replace('"', "'"), largura)) or "?"


def esquema_societario(op: str, info, ac: pd.DataFrame, fre: pd.DataFrame, bruto: pd.DataFrame, niveis: int = 3,
                       pct_min: float = 1.0) -> tuple[str, str]:
    """DOT (graphviz) de quem investe em quem: acionistas -> SPE, em níveis quando o FRE traz a cadeia de controle.
    Fontes: Formulário de Referência (posição acionária, item 15.1; colunas de ID do acionista e do acionista
    relacionado) e a curadoria do cadastro. Linhas 'Ações Tesouraria'/'Outros' e participações zeradas ficam de fora."""
    arestas, nota = [], []
    b = pd.DataFrame()
    if not bruto.empty and "ids" not in bruto:
        cnpjs = set(fre["cnpj"]) if len(fre) else set()
        b = bruto[bruto["_cnpj"].isin(cnpjs)].copy() if "_cnpj" in bruto and cnpjs else pd.DataFrame()
    if len(b):
        col = lambda *ps, ex=(): next((c for c in b.columns if all(p in _n(c) for p in ps)  # noqa: E731
                                       and not any(x in _n(c) for x in ex)), None)
        c_nome = col("acionista", ex=("cpf", "cnpj", "tipo", "id", "controlador", "residente", "relacion", "acordo"))
        c_id = col("id", "acionista", ex=("relacion",))
        c_pai = col("relacion")
        c_pct = col("percentual", "total")
        c_ctrl = col("controlador")
        if c_nome and c_id and c_pai and c_pct:
            b["pct"] = pd.to_numeric(b[c_pct].str.replace(",", "."), errors="coerce")
            b = b[~b[c_nome].map(_n).str.contains(r"tesouraria|^outros$", regex=True) & (b["pct"].fillna(0) > 0)]
            if "ano_fre" in b:
                b = b[b["ano_fre"] == b["ano_fre"].max()]
            nomes_id = dict(zip(b[c_id].astype(str), b[c_nome]))
            pai_de = {str(i): str(p_).strip() for i, p_ in zip(b[c_id], b[c_pai])}

            def nivel(i, k=0):
                p_ = pai_de.get(i, "")
                return 1 if p_ in ("", "nan", "None") or p_ not in pai_de or k > 10 else 1 + nivel(p_, k + 1)
            b["_nv"] = [nivel(str(i)) for i in b[c_id]]
            incluidos: set[str] = set()
            for _, r in b.sort_values("_nv").iterrows():
                i = str(r[c_id])
                nv = r["_nv"]
                pai_id = pai_de.get(i, "")
                if nv > niveis or (r["pct"] < pct_min and nv > 1) or (nv > 1 and pai_id not in incluidos):
                    continue
                incluidos.add(i)
                destino = nomes_id.get(pai_id)
                arestas.append((r[c_nome], destino or info["nome"], r["pct"],
                                str(r[c_ctrl]).upper().startswith("S") if c_ctrl else False))
            nota.append("Formulário de Referência (posição acionária, cadeia de controle informada pela SPE)")
    if not arestas and len(fre):
        f2 = fre[~fre.acionista.map(_n).str.contains(r"tesouraria|^outros$", regex=True) & (fre.participacao_pct.fillna(0) > 0)]
        for _, r in f2.iterrows():
            arestas.append((r.acionista, info["nome"], r.participacao_pct, str(r.controlador).upper().startswith("S")))
        nota.append("Formulário de Referência (posição acionária; sem a ligação entre níveis, todos aparecem ligados à SPE — "
                    "inclui acionistas dos controladores)")
    if len(ac):
        for _, r in ac.iterrows():
            pct = pd.to_numeric(r.get("participacao_pct"), errors="coerce")
            if not any(_n(a_[0]) == _n(r.acionista) for a_ in arestas):
                arestas.append((r.acionista, info["nome"], pct, "control" in _n(r.get("papel", ""))))
        nota.append("curadoria do cadastro (marcados 'validar' sem documento)")
    if not arestas:
        return "", ""
    cor_g = COR_GRUPO.get(info["grupo"], "#1f77b4")
    linhas = ['digraph G {', 'rankdir=TB; bgcolor="transparent"; nodesep=0.25; ranksep=0.45;',
              'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, color="#888888", '
              'fillcolor="#F4F4F4", fontcolor="#222222"];', 'edge [color="#888888", fontname="Helvetica", fontsize=9, fontcolor="#555555"];',
              f'"{_rot(info["nome"])}" [fillcolor="{cor_g}", fontcolor="white", penwidth=2, fontsize=12];']
    vistos = set()
    for orig, dest, pct, ctrl in arestas:
        o, d_ = _rot(orig), _rot(dest)
        if (o, d_) in vistos or o == d_:
            continue
        vistos.add((o, d_))
        if ctrl:
            linhas.append(f'"{o}" [color="{cor_g}", penwidth=2];')
        lab = f'{pct:.1f}%'.replace(".", ",") if pd.notna(pct) else "?"
        linhas.append(f'"{o}" -> "{d_}" [label="{lab}"{", penwidth=2" if ctrl else ""}];')
    linhas.append(f'labelloc="t"; label="Grupo: {_rot(info["grupo"], 60)}"; fontname="Helvetica"; fontsize=11;')
    linhas.append("}")
    return "\n".join(linhas), ("Fonte: " + "; ".join(nota) + ". Setas = participação no capital total; borda grossa = "
                               "controlador. Esquema montado aqui (cálculo próprio).")


# ================================================================== tráfego (cálculo próprio)
def metricas(cad: pd.DataFrame, traf: pd.DataFrame) -> pd.DataFrame:
    mens = traf.groupby(["id_concessao", "mes"], as_index=False)[["veic_eq", "veic_total", "veic_pesado"]].sum()
    linhas = []
    for idc, g in mens.groupby("id_concessao"):
        g = g.sort_values("mes").set_index("mes")
        ult = g.index.max()
        ltm = g.loc[g.index > ult - pd.DateOffset(months=12)]
        ant = g.loc[(g.index > ult - pd.DateOffset(months=24)) & (g.index <= ult - pd.DateOffset(months=12))]
        ltm3 = g.loc[(g.index > ult - pd.DateOffset(months=48)) & (g.index <= ult - pd.DateOffset(months=36))]
        completo = len(ltm) == 12
        eq = ltm.veic_eq.sum()
        cresc = eq / ant.veic_eq.sum() - 1 if completo and len(ant) == 12 and ant.veic_eq.sum() > 0 else np.nan
        cagr3 = (eq / ltm3.veic_eq.sum()) ** (1 / 3) - 1 if completo and len(ltm3) == 12 and ltm3.veic_eq.sum() > 0 else np.nan
        tp = traf[traf.id_concessao == idc]
        j1 = tp[tp.mes > ult - pd.DateOffset(months=12)]
        j0 = tp[(tp.mes > ult - pd.DateOffset(months=24)) & (tp.mes <= ult - pd.DateOffset(months=12))]
        ok1 = j1.groupby("praca").mes.nunique().loc[lambda s: s == 12].index
        ok0 = j0.groupby("praca").mes.nunique().loc[lambda s: s == 12].index
        comuns = ok1.intersection(ok0)
        base0 = j0[j0.praca.isin(comuns)].veic_eq.sum()
        cresc_mp = j1[j1.praca.isin(comuns)].veic_eq.sum() / base0 - 1 if len(comuns) and base0 > 0 else np.nan
        anual = g.veic_eq.groupby(g.index.year).agg(["sum", "count"])
        anual = anual[anual["count"] == 12]["sum"]
        pior = anual.pct_change().min() if len(anual) > 1 else np.nan
        linhas.append({"id_concessao": idc, "ultimo_mes": ult, "meses_ltm": len(ltm),
                       "veic_eq_ltm": eq if completo else np.nan, "cresc_ltm": cresc,
                       "cresc_ltm_mesmas_pracas": cresc_mp,
                       "pracas_novas_ltm": len(set(j1.praca) - set(j0.praca)) if len(j0) else np.nan,
                       "cagr_3a": cagr3,
                       "pct_pesados": ltm.veic_pesado.sum() / ltm.veic_total.sum() if ltm.veic_total.sum() else np.nan,
                       "pior_queda_anual": pior})
    m = cad.merge(pd.DataFrame(linhas), left_on="id", right_on="id_concessao", how="left")
    km = pd.to_numeric(m.km_cadastro, errors="coerce")
    m["vdm_eq_por_km"] = m.veic_eq_ltm / 365 / km
    m["regiao_nome"] = m.regiao.map(REGIOES)
    return m


def anual_antt(nome: str) -> pd.DataFrame:
    d = ler(nome)
    if d.empty:
        return d
    return d[d.id_concessao != ""].groupby(["id_concessao", "ano"], as_index=False)["valor"].sum()


# ================================================================== utilidades de tela
def excel_bytes(abas: dict) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for nome, df in abas.items():
            df.to_excel(w, sheet_name=str(nome)[:31], index=False)
    return buf.getvalue()


def baixar_botao(df: pd.DataFrame, nome: str) -> None:
    c1, c2 = st.columns(2)
    c1.download_button("Baixar CSV", df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"), f"{nome}.csv",
                       key=f"csv_{nome}")
    c2.download_button("Baixar XLSX", excel_bytes({"dados": df}), f"{nome}.xlsx", key=f"xlsx_{nome}")


def legenda(cores: dict, titulo: str = "Grupos") -> None:
    chips = "".join(f'<span style="display:inline-flex;align-items:center;margin:2px 12px 2px 0;font-size:0.85rem">'
                    f'<span style="width:12px;height:12px;border-radius:50%;background:{c};display:inline-block;'
                    f'margin-right:6px"></span>{g}</span>' for g, c in cores.items())
    st.markdown(f"<div><b>{titulo}:</b> {chips}</div>", unsafe_allow_html=True)


def hex_rgb(h: str) -> list[int]:
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) for i in (0, 2, 4)]


def fmt_mm(v) -> str:
    return "-" if pd.isna(v) else f"R$ {v:,.0f} mm".replace(",", ".")


def fmt_pct(v, casas=1) -> str:
    return "-" if pd.isna(v) else f"{v * 100:.{casas}f}%".replace(".", ",")


def layout(fig, titulo, h=360, pct=False, x=False):
    fig.update_layout(title=titulo, height=h, legend=dict(orientation="h", y=-0.22), margin=dict(t=45, b=10, l=10, r=10))
    if pct:
        fig.update_yaxes(tickformat=".0%")
    if x:
        fig.update_xaxes(type="category")
    return fig


def periodo_rotulo(df: pd.DataFrame) -> pd.Series:
    return np.where(df["tipo"] == "anual", df["exercicio"].astype(str), df["periodo"].astype(str))


ROT = {"veic_eq": "Veículos equivalentes", "receita_aj": "Receita líquida ajustada (R$ mm)",
       "ebitda_aj": "EBITDA ajustado (R$ mm)", "ebitda_aj_pos_arrend": "EBITDA ajustado pós-arrendamento (R$ mm)",
       "divida_bruta": "Dívida bruta (R$ mm)", "divida_liquida": "Dívida líquida (R$ mm)",
       "capex": "Capex (R$ mm)", "capex_manut": "Capex de manutenção (R$ mm)",
       "margem_ebitda_aj": "Margem EBITDA ajustada", "icsd": "ICSD (1) regra geral, com capex de manutenção",
       "icsd_arr": "ICSD (2) arrend. no denominador, com capex de manutenção",
       "icsd_arr_sem_manut": "ICSD (2) arrend. no denominador, sem provisão",
       "icsd_sem_manut": "ICSD (1) regra geral, sem provisão", "dl_ebitda_aj_ifrs16": "DL c/ arrend. ÷ EBITDA aj.",
       "dl_ebitda_aj_pre_ifrs16": "DL s/ arrend. ÷ EBITDA aj. pós-arrend.", "custo_implicito": "Custo implícito da dívida",
       "capex_por_veq": "Capex por veíc. eq. (R$)", "capex_manut_por_veq": "Capex manut. por veíc. eq. (R$)",
       "capex_por_km": "Capex por km (R$ mm)", "capex_manut_por_km": "Capex manut. por km (R$ mm)",
       "tarifa_media": "Tarifa média por veíc. eq. (R$)", "ebitda_aj_por_km": "EBITDA aj. por km (R$ mm)",
       "pmr_dias": "PMR (dias)", "pmp_dias": "PMP (dias)", "ciclo_financeiro_dias": "Ciclo financeiro (dias)",
       "pct_divida_cp": "% da dívida no curto prazo", "fco_comp_pos_arrend": "FCO comparável pós-arrendamento (R$ mm)",
       "conversao_fco": "Conversão: FCO comp. pós-arrend. ÷ EBITDA aj. pós-arrend.", "veic_eq_ltm": "Veíc. eq. LTM", "cresc_ltm": "Crescimento LTM",
       "cresc_ltm_mesmas_pracas": "Crescimento LTM (mesmas praças)", "cagr_3a": "CAGR 3 anos", "pct_pesados": "% pesados",
       "vdm_eq_por_km": "VDM eq. por km", "pior_queda_anual": "Pior queda anual", "pracas_novas_ltm": "Praças novas LTM"}
PCT = {"margem_ebitda_aj", "custo_implicito", "pct_divida_cp", "cresc_ltm", "cresc_ltm_mesmas_pracas", "cagr_3a",
       "pct_pesados", "pior_queda_anual", "margem_ebitda_aj_pos_arrend", "conversao_fco"}


# ================================================================== dados
cad = ler_cadastro()
traf = trafego_mensal()
if cad.empty or traf.empty:
    st.error("Não foi possível carregar os dados do coletor. Tente novamente em alguns minutos.")
    st.stop()
met = metricas(cad, traf)
fin = financeiro(cad, traf)
spe_cvm = ler_fin("spe_cvm")
nomes = dict(zip(cad.id, cad.nome))
GRUPOS = sorted(cad.grupo.unique(), key=lambda g: (g == SEM_GRUPO, g))
COR_GRUPO = {g: PALETA[i % len(PALETA)] for i, g in enumerate(GRUPOS)}
COR_GRUPO[SEM_GRUPO] = "#B0B0B0"

# ================================================================== barra lateral
st.sidebar.title("🛣️ BR Toll Roads Fin Data")
st.sidebar.caption(f"Tráfego até {traf.mes.max():%m/%Y} · {len(cad)} contratos no cadastro · "
                   + (f"financeiro CVM de {fin['id'].nunique()} SPEs (último período {fin['dt_fim'].max():%m/%Y})"
                      if not fin.empty else "financeiro CVM ainda não coletado"))
st.sidebar.header("Filtros")
f_ag = st.sidebar.multiselect("Agência", sorted(cad.agencia.unique()), default=sorted(cad.agencia.unique()))
f_reg = st.sidebar.multiselect("Região", list(REGIOES.values()), default=list(REGIOES.values()))
f_gr = st.sidebar.multiselect("Grupo controlador", GRUPOS)
so_ativas = st.sidebar.checkbox("Esconder SPEs sem tráfego nos últimos 12 meses", value=True,
                                help="SPEs sem nenhuma série de tráfego publicada (ex.: Artemig) continuam na lista.")
sel = met[met.agencia.isin(f_ag) & met.regiao_nome.isin(f_reg)]
if f_gr:
    sel = sel[sel.grupo.isin(f_gr)]
if so_ativas:
    corte = traf.mes.max() - pd.DateOffset(months=12)
    sel = sel[(sel.ultimo_mes > corte) | sel.ultimo_mes.isna()]
st.sidebar.caption(f"{len(sel)} SPEs no filtro")
st.sidebar.markdown("---")
st.sidebar.caption("Price Discovery · dados abertos ANTT, ARTESP, CVM, SND, ANBIMA, IBGE e BCB. "
                   "Grupo marcado 'validar' vem de conhecimento prévio, ainda sem documento.")

st.title("BR Toll Roads Fin Data")
aba_mapa, aba_op, aba_comp, aba_jur, aba_mac, aba_met = st.tabs(
    ["Visão geral e mapa", "Operacional por SPE", "Comparar SPEs", "Juros (ETTJ)", "Macro (inflação, PIB, câmbio)", "Metodologia"])
pj = carregar_juros()


def fin_filtrado(ids, tipo="anual") -> pd.DataFrame:
    if fin.empty:
        return fin
    return fin[fin["id"].isin(ids) & (fin["tipo"] == tipo)]


# ================================================================== visão geral e mapa
with aba_mapa:
    c = st.columns(4)
    c[0].metric("SPEs no filtro", len(sel))
    c[1].metric("Veíc. equivalentes LTM (mi)", f"{sel.veic_eq_ltm.sum() / 1e6:,.0f}")
    c[2].metric("Grupos", sel.grupo.nunique())
    c[3].metric("Último mês com dado", traf.mes.max().strftime("%m/%Y"))
    pr = ler("pracas_antt")
    pontos = []
    ltm_praca = traf[traf.mes > traf.mes.max() - pd.DateOffset(months=12)].groupby(["id_concessao", "praca"]).veic_eq.sum()
    com_praca = set()
    if not pr.empty:
        pr = pr[pr.id_concessao.isin(sel.id) & pr.latitude.notna()].merge(
            cad[["id", "nome", "grupo"]], left_on="id_concessao", right_on="id")
        pr["veic_eq_ltm"] = [ltm_praca.get((i, p), 0) for i, p in zip(pr.id_concessao, pr.praca_de_pedagio)]
        pr["local"] = pr.praca_de_pedagio + " — " + pr.rodovia.astype(str) + "/" + pr.uf.astype(str)
        com_praca = set(pr.id_concessao)
        pontos.append(pr[["nome", "grupo", "local", "latitude", "longitude", "veic_eq_ltm"]])
    # SPEs sem praça georreferenciada: ponto aproximado do cadastro (sede/centro do trecho) — só quando não há traçado
    # oficial; com traçado (ex.: lotes ARTESP), a linha já localiza a SPE e o VEQ LTM vai no tooltip da linha
    tracado = st.checkbox("Mostrar traçado das rodovias", value=True, key="map_tracado")
    # a chave do cache inclui data/tamanho do arquivo: cada atualização do workflow aparece sem esperar o TTL
    _st = ARQ_TRACADOS.stat() if ARQ_TRACADOS.exists() else None
    tr_of, versao_snv = tracados_oficiais((_st.st_mtime_ns, _st.st_size) if _st else ())
    ids_tracado = set(tr_of.id_concessao) if (tracado and not tr_of.empty) else set()
    sem = sel[~sel.id.isin(com_praca) & ~sel.id.isin(ids_tracado)].copy()
    sem["latitude"], sem["longitude"] = pd.to_numeric(sem.lat, errors="coerce"), pd.to_numeric(sem.lon, errors="coerce")
    sem = sem.dropna(subset=["latitude"])
    if len(sem):
        sem["local"] = "posição aproximada do trecho (sem praça georreferenciada)"
        sem["veic_eq_ltm"] = sem.veic_eq_ltm.fillna(0)
        pontos.append(sem[["nome", "grupo", "local", "latitude", "longitude", "veic_eq_ltm"]])
    if pontos:
        pt = pd.concat(pontos, ignore_index=True)
        pt["cor"] = pt.grupo.map(lambda g: hex_rgb(COR_GRUPO.get(g, "#B0B0B0")))
        pt["raio"] = 4000 + np.sqrt(pt.veic_eq_ltm.astype(float)) * 4
        pt["veic_eq_ltm_txt"] = pt.veic_eq_ltm.map(lambda v: f"{v:,.0f}".replace(",", "."))
        camadas = []
        ufs_geo, ufs_rot = estados_geojson()
        if ufs_geo:
            camadas.append(pdk.Layer("GeoJsonLayer", ufs_geo, stroked=True, filled=True, get_fill_color=[128, 128, 128, 18],
                                     get_line_color=[160, 160, 160, 220], line_width_min_pixels=1, pickable=False))
            camadas.append(pdk.Layer("TextLayer", ufs_rot, get_position="[lon, lat]", get_text="sigla", get_size=12,
                                     get_color=[150, 150, 150, 230], pickable=False))
        com_tracado = set()
        if tracado:
            caminhos = []
            info_sel = sel.set_index("id")
            if not tr_of.empty:
                t_ = tr_of[tr_of.id_concessao.isin(sel.id)].copy()
                com_tracado = set(t_.id_concessao)
                for c_ in ("nome", "grupo", "veic_eq_ltm"):
                    t_[c_] = t_.id_concessao.map(info_sel[c_])
                t_["local"] = "Traçado: " + t_.trecho + t_.id_concessao.map(
                    lambda i: "\n⚠ traçado a validar (ver nota abaixo do mapa)" if i in NOTA_TRACADO else "")
                caminhos.append(t_)
            # reserva para SPE sem traçado oficial: linha ligando as praças da mesma rodovia/UF na ordem do km
            if not pr.empty:
                q = pr[~pr.id_concessao.isin(com_tracado)].assign(km=pd.to_numeric(pr["km_m"], errors="coerce")).dropna(subset=["km"])
                aprox = []
                for (_i, _rod, _uf), gq in q.groupby(["id_concessao", "rodovia", "uf"]):
                    gq = gq.drop_duplicates(["latitude", "longitude"]).sort_values("km")
                    if len(gq) >= 2:
                        aprox.append({"id_concessao": _i, "path": gq[["longitude", "latitude"]].astype(float).values.tolist(),
                                      "nome": gq.nome.iloc[0], "grupo": gq.grupo.iloc[0],
                                      "veic_eq_ltm": info_sel.veic_eq_ltm.get(_i, 0),
                                      "local": f"Aproximação (liga as praças): {_rod}/{_uf}"})
                if aprox:
                    caminhos.append(pd.DataFrame(aprox))
            if caminhos:
                cm = pd.concat(caminhos, ignore_index=True)
                cm["cor"] = cm.grupo.map(lambda g: hex_rgb(COR_GRUPO.get(g, "#B0B0B0")))
                cm["veic_eq_ltm_txt"] = pd.to_numeric(cm.veic_eq_ltm, errors="coerce").fillna(0).map(
                    lambda v: f"{v:,.0f}".replace(",", "."))
                camadas.append(pdk.Layer("PathLayer", cm[["path", "cor", "nome", "grupo", "local", "veic_eq_ltm_txt"]],
                                         get_path="path", get_color="cor", width_min_pixels=3, get_width=4,
                                         opacity=0.85, pickable=True, auto_highlight=True, joint_rounded=True,
                                         cap_rounded=True))
        st.pydeck_chart(pdk.Deck(
            map_style=None, initial_view_state=pdk.ViewState(latitude=-18, longitude=-50, zoom=3.6),
            layers=camadas + [pdk.Layer("ScatterplotLayer", pt, get_position="[longitude, latitude]", get_fill_color="cor",
                              get_radius="raio", pickable=True, opacity=0.8, stroked=True, get_line_color=[255, 255, 255],
                              line_width_min_pixels=1)],
            tooltip={"text": "{nome} ({grupo})\n{local}\nVeíc. eq. LTM: {veic_eq_ltm_txt}"}))
        legenda({g: COR_GRUPO[g] for g in GRUPOS if g in set(pt.grupo)})
        sem_pos = sel[~sel.id.isin(set(com_praca)) & ~sel.id.isin(set(sem.id)) & ~sel.id.isin(com_tracado)]
        sem_tr = sel[~sel.id.isin(com_tracado)].nome if tracado else pd.Series(dtype=str)
        st.caption("Divisas e siglas dos estados: GeoJSON público (click_that_hood), simplificado. "
                   f"Traçado das federais: km inicial/final de cada trecho concedido (ANTT, dados abertos) cortado sobre a "
                   f"geometria do SNV {versao_snv} do DNIT — só a pista principal das BRs (acessos e rodovias estaduais "
                   "dos lotes do PR não estão no SNV). SP: malha de cada lote nos KMZ da ARTESP (dados abertos). Contratos "
                   "federais encerrados e BRs delegadas a MG: mesmo trecho da sucessora ou pelas descrições de local do SNV "
                   "(critério em geo/complementos_tracados.csv — cálculo próprio). Sem traçado oficial, a linha liga as "
                   "praças na ordem do km (aproximação). Círculos = praças federais (coordenadas da ANTT), tamanho = veículos "
                   "equivalentes que passaram na praça nos últimos 12 meses. SPEs sem praça georreferenciada (SP e MG) "
                   "aparecem só pela linha — o VEQ LTM da SPE fica no tooltip; sem traçado, num ponto aproximado do cadastro."
                   + (f" Sem traçado: {', '.join(sem_tr)}." if len(sem_tr) else "")
                   + (f" Sem posição no mapa: {', '.join(sem_pos.nome)}." if len(sem_pos) else ""))
        notas = [NOTA_TRACADO[i] for i in NOTA_TRACADO if i in set(sel.id)]
        if tracado and notas:
            st.caption("⚠ **Traçados a validar** — " + " ".join(notas))

    # ---- consolidado (gráficos)
    st.subheader("Consolidado das SPEs no filtro")
    por_spe = len(f_gr) >= 1
    cor = "nome" if por_spe else "grupo"
    base_per = st.radio("Período", ["Anos fechados", "Anos fechados + últimos 12 meses (LTM)"], index=1, horizontal=True,
                        key="map_periodo",
                        help="LTM = últimos 12 meses até o último ITR (soma dos 4 trimestres mais recentes, montada a partir "
                             "das DFP/ITR da CVM); para o tráfego, os 12 meses até o último mês publicado. Fica como a "
                             "última barra, depois dos anos fechados. Saldos (dívida) = posição na data de fim do período.")
    com_ltm = base_per.endswith("(LTM)")
    st.caption("Empilhado por " + ("SPE (grupo selecionado no filtro)" if por_spe else "grupo controlador")
               + ". Fontes: tráfego ANTT/ARTESP; financeiro CVM (DFP e, no LTM, ITR) das SPEs com registro. "
                 "Somas de SPEs, não o consolidado contábil da holding (sem eliminações nem dívida da holding).")

    # financeiro: anos fechados (ano da data de fim do exercício) + LTM opcional
    ft = fin_filtrado(sel.id, "anual").copy()
    rot_ltm_fin, ltm_atras = None, []
    if not ft.empty:
        ft["ano"] = ft.dt_fim.dt.year
        ft = ft.sort_values("dt_fim").drop_duplicates(["id", "ano"], keep="last")
        ft = ft[ft.ano >= ft.ano.max() - 9]
        ft["periodo_x"], ft["ordem"] = ft.ano.astype(str), ft.ano.astype(float)
        if com_ltm:
            fl = fin_filtrado(sel.id, "ltm").copy()
            if not fl.empty and fl.dt_fim.max() > ft.dt_fim.max():
                dt_ltm = fl.dt_fim.max()
                fl = fl.sort_values("dt_fim").groupby("id").tail(1)
                fl = fl[fl.dt_fim > dt_ltm - pd.DateOffset(months=12)]
                ltm_atras = [f"{n} ({d:%m/%Y})" for n, d in zip(fl.nome, fl.dt_fim) if d < dt_ltm]
                rot_ltm_fin = f"LTM {dt_ltm:%m/%Y}"
                fl["ano"], fl["periodo_x"], fl["ordem"] = np.nan, rot_ltm_fin, 9999.0
                ft = pd.concat([ft, fl], ignore_index=True)
    # tráfego: anos completos (12 meses) + LTM opcional
    tr = traf[traf.id_concessao.isin(sel.id)].copy()
    tr["ano"] = tr.mes.dt.year
    cont = tr.groupby(["id_concessao", "ano"]).mes.nunique()
    ta = tr.groupby(["id_concessao", "ano"], as_index=False).veic_eq.sum()
    ta = ta[ta.set_index(["id_concessao", "ano"]).index.map(cont) == 12]
    ta = ta[ta.ano >= ta.ano.max() - 9] if len(ta) else ta
    ta["periodo_x"], ta["ordem"] = ta.ano.astype(str), ta.ano.astype(float)
    rot_ltm_tr = None
    mes_max = traf.mes.max()
    if com_ltm and mes_max.month != 12:
        jan = tr[tr.mes > mes_max - pd.DateOffset(months=12)]
        nm = jan.groupby("id_concessao").mes.nunique()
        tl = jan[jan.id_concessao.map(nm) == 12].groupby("id_concessao", as_index=False).veic_eq.sum()
        rot_ltm_tr = f"LTM {mes_max:%m/%Y}"
        tl["ano"], tl["periodo_x"], tl["ordem"] = np.nan, rot_ltm_tr, 9999.0
        ta = pd.concat([ta, tl], ignore_index=True)
    ta = ta.merge(cad[["id", "nome", "grupo"]], left_on="id_concessao", right_on="id")

    def ordem_x(df_):
        return df_.drop_duplicates("periodo_x").sort_values("ordem")["periodo_x"].tolist()

    def consolidado_p(f, col, titulo):
        if f.empty or f[col].notna().sum() == 0:
            st.caption(f"{titulo}: sem dados.")
            return
        g = f.groupby(["periodo_x", cor], as_index=False)[col].sum(min_count=1)
        fig = px.bar(g, x="periodo_x", y=col, color=cor, color_discrete_map=COR_GRUPO if cor == "grupo" else None,
                     category_orders={"periodo_x": ordem_x(f)}, labels={"periodo_x": "", col: "", cor: ""})
        st.plotly_chart(layout(fig, titulo, x=True), width="stretch")

    k1, k2 = st.columns(2)
    with k1:
        consolidado_p(ta, "veic_eq", "Veículos equivalentes (anos completos" + (f" + {rot_ltm_tr})" if rot_ltm_tr else ")"))
    if ft.empty:
        k2.info("Dados financeiros da CVM ainda não coletados: rode o workflow financeiro-spes no coletor.")
    else:
        with k2:
            consolidado_p(ft, "receita_aj", "Receita líquida ajustada (sem construção, R$ mm)")
        k1, k2 = st.columns(2)
        with k1:
            consolidado_p(ft, "ebitda_aj", "EBITDA ajustado (+ D&A + provisão de manutenção, R$ mm)")
        with k2:
            consolidado_p(ft, "capex", "Capex (imobilizado + intangível + ativo de contrato, R$ mm)")
        k1, k2 = st.columns(2)
        with k1:
            consolidado_p(ft, "capex_manut", "Capex de manutenção (pagamento da provisão, R$ mm)")
        with k2:
            g = ft.groupby(["periodo_x", "ordem"], as_index=False)[["divida_bruta", "divida_liquida"]].sum(min_count=1)
            g = g.sort_values("ordem")
            fig = go.Figure([go.Bar(x=g.periodo_x, y=g.divida_bruta, name="Dívida bruta"),
                             go.Bar(x=g.periodo_x, y=g.divida_liquida, name="Dívida líquida")])
            fig = layout(fig, "Dívida bruta e líquida (ex-arrendamento, fim do período, R$ mm)", x=True)
            fig.update_xaxes(categoryorder="array", categoryarray=g.periodo_x.tolist())
            st.plotly_chart(fig, width="stretch")
        if por_spe:
            consolidado_p(ft, "divida_bruta", "Dívida bruta por SPE (R$ mm)")
        if rot_ltm_fin:
            st.caption(f"{rot_ltm_fin}: SPEs com ITR até {ft.dt_fim.max():%m/%Y}."
                       + (f" Com último ITR anterior (usado o LTM mais recente de cada uma): {', '.join(ltm_atras)}."
                          if ltm_atras else "")
                       + " SPEs sem ITR nos últimos 12 meses ficam fora da barra LTM.")

    # ---- participação: % por período, com seleção do indicador (cálculo próprio)
    st.markdown("**Participação por período** (cálculo próprio)")
    vars_part = ["veic_eq", "receita_aj", "ebitda_aj", "capex", "capex_manut", "divida_bruta", "divida_liquida"]
    var_p = st.selectbox("Indicador", vars_part, format_func=ROT.get, key="map_part_ind")
    src = ta if var_p == "veic_eq" else ft
    if src.empty or var_p not in src or src[var_p].notna().sum() == 0:
        st.caption("Sem dados para o indicador no filtro.")
    else:
        chave = ["periodo_x", "ordem", "grupo"] + (["nome"] if por_spe else [])
        g = src.groupby(chave, as_index=False)[var_p].sum(min_count=1).dropna(subset=[var_p])
        tot = g.groupby(["periodo_x", "grupo"] if por_spe else ["periodo_x"])[var_p].transform("sum")
        g["part"] = g[var_p] / tot.where(tot > 0)
        g = g.dropna(subset=["part"])
        n_gr = g.grupo.nunique() if por_spe else 1
        fig = px.bar(g, x="periodo_x", y="part", color=cor, color_discrete_map=COR_GRUPO if cor == "grupo" else None,
                     facet_row="grupo" if por_spe and n_gr > 1 else None, text_auto=".0%",
                     category_orders={"periodo_x": ordem_x(g)}, labels={"periodo_x": "", "part": "", cor: "", "grupo": ""},
                     hover_data={var_p: ":,.0f", "part": ":.1%"})
        fig.update_traces(textposition="inside", insidetextanchor="middle", textfont_size=10)
        fig = layout(fig, f"{ROT[var_p]}: participação " + ("de cada SPE no total do grupo" if por_spe else
                                                            "de cada grupo no total do filtro"),
                     h=max(420, 300 * n_gr), pct=True, x=True)
        fig.update_layout(barmode="stack", uniformtext_minsize=9, uniformtext_mode="hide")
        st.plotly_chart(fig, width="stretch")
        st.caption("Cada barra soma 100% do período (SPEs com dado no período: DFP nos anos fechados; ITR no LTM; "
                   "tráfego com 12 meses completos). Períodos em que o total é zero ou negativo (possível na dívida "
                   "líquida) ficam de fora; SPE com valor negativo aparece abaixo de zero.")
        with st.expander("Tabela da participação"):
            tb = g.pivot_table(index=["grupo", "nome"] if por_spe else "grupo", columns="periodo_x", values="part")
            tb = tb[[c_ for c_ in ordem_x(g) if c_ in tb.columns]].reset_index()
            st.dataframe(tb, hide_index=True, width="stretch",
                         column_config={c_: st.column_config.NumberColumn(format="percent") for c_ in ordem_x(g)})

    # ---- custo da dívida consolidado (debêntures)
    st.subheader("Custo da dívida e spread sobre o soberano")
    deb, fonte_deb = debentures(cad)
    deb_f = deb[deb["id"].isin(sel.id)].drop_duplicates("codigo") if not deb.empty else deb
    cA, cB = st.columns([2, 1])
    if deb_f.empty:
        cA.info("Sem debêntures das SPEs no filtro na base disponível.")
    else:
        cd = custo_debentures(deb_f, pj)
        r = resumo_custo(cd)
        cA.markdown(f"**Debêntures das SPEs** — fonte: {fonte_deb}; ponderado pelo saldo (cálculo próprio)")
        cA.dataframe(r.round(2), hide_index=True, width="stretch")
        tot = r[r["Indexador"].str.startswith("Total")].iloc[0]
        cB.metric("Custo nominal de emissão (pond.)", f"{tot['Custo nominal emissão (% a.a.)']:.2f}%")
        cB.metric("Custo nominal a mercado (pond.)", f"{tot['Custo nominal mercado (% a.a.)']:.2f}%")
        cB.metric("Spread emissão s/ pré soberano", f"{tot['Spread nominal emissão s/ pré (p.p.)']:.2f} p.p.")
        cB.metric("Spread mercado s/ pré soberano", f"{tot['Spread nominal mercado s/ pré (p.p.)']:.2f} p.p.")
    st.caption("Emissão = remuneração da escritura (custo vigente da dívida existente); mercado = taxa indicativa ANBIMA "
               "(custo de refinanciar hoje). Soberano = ETTJ ANBIMA no prazo (duration): real para IPCA+, pré para "
               "prefixadas; no CDI+ o spread é o próprio spread sobre o CDI. Nominal = custo equivalente com a curva de hoje.")

    cols = ["nome", "agencia", "uf_principal", "regiao_nome", "grupo", "grupo_fonte", "km_cadastro", "modalidade", "lote_obs", "obs"]
    with st.expander("Cadastro das SPEs no filtro"):
        st.dataframe(sel[cols], hide_index=True, width="stretch")

# ================================================================== operacional por SPE
with aba_op:
    op = st.selectbox("SPE", sel.id.tolist() or cad.id.tolist(),
                      format_func=lambda i: f"{nomes.get(i, i)} ({cad.set_index('id').grupo.get(i, '')})")
    info = cad.set_index("id").loc[op]
    reg = spe_cvm.set_index("id").loc[op] if (not spe_cvm.empty and op in set(spe_cvm.id)) else None
    st.subheader(f"{info['nome']} — {info['grupo']}")
    st.caption(f"{info['agencia']} · {info['uf_principal']} · {info['modalidade']} · km (cadastro): {info['km_cadastro'] or '-'}"
               + (f" · {info['lote_obs']}" if info['lote_obs'] else "") + (f" · {info['obs']}" if info['obs'] else ""))
    if "PPP" in str(info["modalidade"]):
        st.warning("Concessão patrocinada (PPP): contraprestação do poder concedente muda a contabilidade (ativo "
                   "financeiro/de contrato) e as métricas de receita e capex — comparar com concessões comuns com cautela.")

    t = traf[traf.id_concessao == op]
    m = pd.DataFrame()
    pasta_m = pasta_macro()
    st.markdown("### Tráfego")
    if t.empty:
        st.info("Sem tráfego publicado para esta SPE.")
    else:
        fonte = t.fonte.iloc[0]
        m = t.groupby("mes", as_index=False)[["veic_eq", "veic_total", "veic_pesado"]].sum().sort_values("mes")
        m["ltm"] = m.veic_eq.rolling(12).sum()
        pm = pracas_mensal(t)                     # praça x mês (calendário completo)
        n_pracas = pm.shape[1]
        ult = m.mes.max()

        # ---------- volume e distribuição por praça (mesma frequência)
        st.markdown("#### Volume")
        freq = st.radio("Frequência", ["Mensal", "Trimestral", "Últimos 12 meses (LTM)", "Anual (anos completos)"],
                        horizontal=True, key="op_freq",
                        help="Vale para os dois gráficos abaixo. Trimestral = soma dos 3 meses de cada trimestre (só "
                             "trimestres completos), sem acumular. LTM = soma móvel dos últimos 12 meses, um ponto por mês "
                             "(tira a sazonalidade). Anual = só anos com 12 meses publicados.")
        if freq == "Mensal":
            fig = go.Figure([go.Bar(x=m.mes, y=m.veic_eq, name="Veíc. equivalentes"),
                             go.Bar(x=m.mes, y=m.veic_total, name="Veíc. físicos", opacity=0.5)])
            fig.update_layout(barmode="group")
            dist = pm.dropna(how="all")
            dist.index = list(dist.index)
        elif freq == "Trimestral":
            tq = m.assign(tri=m.mes.dt.to_period("Q")).groupby("tri").agg(
                veic_eq=("veic_eq", "sum"), veic_total=("veic_total", "sum"), n=("mes", "count"))
            tq = tq[tq.n == 3].reset_index()
            tq["rot"] = [f"{p.quarter}T{str(p.year)[2:]}" for p in tq.tri]
            fig = go.Figure([go.Bar(x=tq.rot, y=tq.veic_eq, name="Veíc. equivalentes"),
                             go.Bar(x=tq.rot, y=tq.veic_total, name="Veíc. físicos", opacity=0.5)])
            fig.update_layout(barmode="group")
            fig.update_xaxes(type="category", categoryorder="array", categoryarray=tq.rot.tolist())
            dist = pm.dropna(how="all").groupby(pm.dropna(how="all").index.to_period("Q")).sum()
            dist = dist.loc[dist.index.isin(tq.tri)]
            dist.index = [f"{p.quarter}T{str(p.year)[2:]}" for p in dist.index]
        elif freq.startswith("Últimos 12"):
            lt_ = m.assign(ltm_total=m.veic_total.rolling(12).sum()).dropna(subset=["ltm"])
            fig = go.Figure([go.Scatter(x=lt_.mes, y=lt_.ltm, mode="lines", name="Veíc. equivalentes (LTM)", line=dict(width=3)),
                             go.Scatter(x=lt_.mes, y=lt_.ltm_total, mode="lines", name="Veíc. físicos (LTM)",
                                        line=dict(width=2, dash="dot"))])
            dist = pm.rolling(12, min_periods=12).sum().dropna(how="all")
            dist.index = list(dist.index)
        else:
            a = m.assign(ano=m.mes.dt.year).groupby("ano").agg(veic_eq=("veic_eq", "sum"), veic_total=("veic_total", "sum"),
                                                                n=("mes", "count"))
            a = a[a.n == 12].reset_index()
            fig = go.Figure([go.Bar(x=a.ano.astype(str), y=a.veic_eq, name="Veíc. equivalentes"),
                             go.Bar(x=a.ano.astype(str), y=a.veic_total, name="Veíc. físicos", opacity=0.5)])
            fig.update_layout(barmode="group")
            dist = pm.dropna(how="all").groupby(pm.dropna(how="all").index.year).sum()
            dist = dist.loc[dist.index.isin(a.ano)]
            dist.index = [str(x_) for x_ in dist.index]
        v1, v2 = st.columns(2)
        v1.plotly_chart(layout(fig, f"Volume de tráfego (fonte: {fonte})", h=400), width="stretch")
        if n_pracas < 2:
            v2.info("Só uma praça publicada para esta SPE: sem distribuição por praça.")
        elif dist.empty:
            v2.info("Sem períodos completos nesta frequência para a distribuição por praça.")
        else:
            todas_p = v2.checkbox(f"Mostrar todas as {n_pracas} praças", value=n_pracas <= 12, key=f"op_dist_todas_{op}",
                                  help="Desmarcado: as 10 praças com mais tráfego nos últimos 12 meses e o resto em 'Outras praças'.")
            peso = pm[pm.index > ult - pd.DateOffset(months=12)].sum()
            dv = dist if todas_p else agrupar_pracas(dist, peso, 10)
            dv = dv[[c_ for c_ in dv.columns if dv[c_].sum() > 0]]
            dv = dv.div(dv.sum(axis=1).replace(0, np.nan), axis=0)
            if freq.startswith("Últimos 12"):
                fig = go.Figure([go.Scatter(x=dv.index, y=dv[c_], name=str(c_), mode="lines", stackgroup="um",
                                            line=dict(width=0.5), hovertemplate="%{y:.1%}") for c_ in dv.columns])
            else:
                fig = go.Figure([go.Bar(x=dv.index, y=dv[c_], name=str(c_), hovertemplate="%{y:.1%}") for c_ in dv.columns])
                fig.update_layout(barmode="stack", bargap=0.05)
                if freq != "Mensal":
                    fig.update_xaxes(type="category")
            fig.update_yaxes(range=[0, 1])
            v2.plotly_chart(layout(fig, "Distribuição do tráfego por praça (% dos veíc. eq.)", h=400, pct=True), width="stretch")
        st.caption("Distribuição = participação de cada praça nos veículos equivalentes do período (cálculo próprio sobre "
                   f"os dados da {fonte}). Praça que some ou aparece pode ser abertura, desativação ou só troca de nome na "
                   "fonte — a ANTT, por exemplo, renomeia praças e cria pórticos de free flow.")

        # ---------- crescimento: LTM (tira a sazonalidade) com comparações e decomposição por praça
        st.markdown("#### Crescimento")
        lp = pm.rolling(12, min_periods=12).sum()           # LTM por praça
        tot = lp.sum(axis=1, min_count=1)
        g = pd.DataFrame({"mes": tot.index, "ltm": tot.values})
        g["cresc"] = g.ltm / g.ltm.shift(12) - 1           # a/a no calendário completo, depois recortado
        g = g.dropna(subset=["ltm"]).reset_index(drop=True)
        if len(g) < 2:
            st.info("Menos de 13 meses de tráfego publicados: sem série de crescimento em 12 meses acumulados.")
        else:
            L = ltm_por_spe()
            outras = [c_ for c_ in L.columns if c_ != op]
            ids_gr = [] if info["grupo"] == SEM_GRUPO else [i for i in cad[cad.grupo == info["grupo"]].id if i in outras]
            ids_rg = [i for i in cad[cad.regiao == info["regiao"]].id if i in outras]
            c1, c2, c3 = st.columns([1.1, 1.5, 1.7])
            modo_c = c1.radio("Medida", ["a/a (LTM vs LTM anterior)", "Base zero (acumulado desde o início)"],
                              key="op_cresc")
            cmp_pib = c1.checkbox("Comparar com o PIB real", key="op_pib")
            decomp = c1.checkbox("Decompor por praça", key="op_decomp", disabled=n_pracas < 2,
                                 help="Quanto cada praça somou (ou tirou) do crescimento total." if n_pracas >= 2
                                 else "Só uma praça publicada para esta SPE.")
            op_med = {f"Grupo {info['grupo']} (demais SPEs, {len(ids_gr)})": ids_gr,
                      f"Região {REGIOES.get(info['regiao'], info['regiao'])} (demais SPEs, {len(ids_rg)})": ids_rg,
                      f"Todas as SPEs da base (demais, {len(outras)})": outras}
            op_med = {k_: v_ for k_, v_ in op_med.items() if v_}
            meds = c2.multiselect("Comparar com a mediana de", list(op_med), key=f"op_cmp_med_{op}")
            cesta = c3.multiselect("Cesta de SPEs", outras, key=f"op_cesta_{op}",
                                   format_func=lambda i: f"{nomes.get(i, i)} ({cad.set_index('id').grupo.get(i, '')})")
            cesta_modo = c3.radio("Mostrar a cesta como", ["Mediana da cesta", "Cada SPE"], horizontal=True,
                                  key="op_cesta_modo", disabled=not cesta)
            g["rot"] = g.mes.dt.strftime("%m/%Y")
            xs_t = list(g.rot)
            p0, p1 = st.select_slider("Prazo analisado", options=xs_t, value=(xs_t[max(0, len(xs_t) - 61)], xs_t[-1]),
                                      key=f"op_traf_prazo_{op}")
            i0, i1 = xs_t.index(p0), xs_t.index(p1)
            gj = g.iloc[i0:i1 + 1].copy()
            mes0, mes1 = gj.mes.iloc[0], gj.mes.iloc[-1]
            aa = modo_c.startswith("a/a")
            if not aa:
                gj["cresc"] = gj.ltm / gj.ltm.iloc[0] - 1
            anos_t = max((mes1 - mes0).days / 365.25, 1e-9)
            resumo_t = []
            if i1 > i0 and gj.ltm.iloc[0] > 0:
                r_ = gj.ltm.iloc[-1] / gj.ltm.iloc[0]
                resumo_t.append({"Série": "Esta SPE", "SPEs": 1, "Acumulado": r_ - 1, "Ao ano (CAGR)": r_ ** (1 / anos_t) - 1})
            fig = go.Figure([go.Scatter(x=gj.mes, y=gj.cresc, mode="lines", name=f"{info['nome']} (veíc. eq. LTM)",
                                        line=dict(width=3.5, color="#1f77b4"))])

            def serie_pares(ids_):
                """Crescimento de cada SPE do conjunto no recorte (mesma medida da SPE) e o acumulado p0 → p1."""
                Lw = L.reindex(gj.mes)[ids_]
                if aa:
                    cr = (L / L.shift(12) - 1).reindex(gj.mes)[ids_]
                else:
                    b_ = Lw.iloc[0].where(Lw.iloc[0] > 0)
                    cr = Lw / b_ - 1
                ac = (Lw.iloc[-1] / Lw.iloc[0].where(Lw.iloc[0] > 0) - 1).dropna()
                return cr, ac

            def add_med(nome_, ids_, cor, traco):
                cr, ac = serie_pares(ids_)
                n_ = cr.notna().sum(axis=1)
                y_ = cr.median(axis=1).where(n_ > 0)
                if y_.notna().any():
                    fig.add_trace(go.Scatter(x=gj.mes, y=y_, mode="lines", name=nome_, line=dict(color=cor, dash=traco, width=2.5),
                                             customdata=n_, hovertemplate="%{y:.1%} · mediana de %{customdata} SPEs"))
                if len(ac) and i1 > i0:
                    med_ = ac.median()
                    resumo_t.append({"Série": nome_, "SPEs": len(ac), "Acumulado": med_,
                                     "Ao ano (CAGR)": (1 + med_) ** (1 / anos_t) - 1 if med_ > -1 else np.nan})

            cores_med = ["#ff7f0e", "#17becf", "#7f7f7f"]
            for (k_, ids_), cor_ in zip(op_med.items(), cores_med):
                if k_ in meds:
                    add_med("Mediana — " + k_.split(" (")[0], ids_, cor_, "dash")
            if cesta:
                if cesta_modo == "Mediana da cesta":
                    add_med(f"Mediana — cesta ({len(cesta)} SPEs)", cesta, "#d62728", "dashdot")
                else:
                    cr, ac = serie_pares(cesta)
                    for j_, i_ in enumerate(cesta):
                        fig.add_trace(go.Scatter(x=gj.mes, y=cr[i_], mode="lines", name=nomes.get(i_, i_),
                                                 line=dict(width=1.6, color=PALETA[(j_ + 3) % len(PALETA)])))
                        if i_ in ac.index and i1 > i0:
                            resumo_t.append({"Série": nomes.get(i_, i_), "SPEs": 1, "Acumulado": ac[i_],
                                             "Ao ano (CAGR)": (1 + ac[i_]) ** (1 / anos_t) - 1 if ac[i_] > -1 else np.nan})
            if cmp_pib:
                pv = pd.read_csv(pasta_m / "pib_volume_trimestral.csv") if (pasta_m / "pib_volume_trimestral.csv").exists() else pd.DataFrame()
                if len(pv):
                    pv["data"] = [pd.Timestamp(a, 3 * q, 1) for a, q in zip(pv.ano, pv.tri)]
                    pv["acum4"] = pv.indice_volume.rolling(4).sum()
                    pv = pv.dropna(subset=["acum4"])
                    pv["cresc"] = pv.acum4.pct_change(4)
                    pv = pv[(pv.data >= mes0 - pd.DateOffset(months=2)) & (pv.data <= mes1)].copy()
                    if len(pv):
                        if not aa:
                            pv["cresc"] = pv.acum4 / pv.acum4.iloc[0] - 1
                        fig.add_trace(go.Scatter(x=pv.data, y=pv.cresc, mode="lines+markers", name="PIB real (4 tri acum.)",
                                                 line=dict(color="#2CA02C", dash="dot", width=2.5)))
                        if len(pv) > 1 and pv.acum4.iloc[0] > 0:
                            r_ = pv.acum4.iloc[-1] / pv.acum4.iloc[0]
                            anos_p = max((pv.data.iloc[-1] - pv.data.iloc[0]).days / 365.25, 1e-9)
                            resumo_t.append({"Série": f"PIB real ({pv.data.iloc[0]:%m/%Y} → {pv.data.iloc[-1]:%m/%Y})", "SPEs": np.nan,
                                             "Acumulado": r_ - 1, "Ao ano (CAGR)": r_ ** (1 / anos_p) - 1})
            fig.add_hline(y=0, line_dash="dot", line_color="gray")
            c_t1, c_t2 = st.columns([1.1, 2.4])
            c_t2.plotly_chart(layout(fig, f"Crescimento do tráfego {p0} → {p1} — " + ("a/a, 12 meses acumulados" if aa
                                                                                   else "acumulado desde o início do prazo"),
                                     h=420, pct=True), width="stretch")
            if resumo_t:
                c_t1.markdown(f"**{p0} → {p1}** ({anos_t:.1f} anos)")
                c_t1.dataframe(pd.DataFrame(resumo_t), hide_index=True, width="stretch",
                               column_config={"Série": st.column_config.TextColumn(width="large"),
                                              "SPEs": st.column_config.NumberColumn(format="%d"),
                                              "Acumulado": st.column_config.NumberColumn(format="percent"),
                                              "Ao ano (CAGR)": st.column_config.NumberColumn(format="percent")})
            st.caption("Escolha início e fim na barra 'Prazo analisado'. Tráfego em 12 meses acumulados (neutraliza a "
                       "sazonalidade); mês sem dado publicado derruba as somas de 12 meses que o contêm. Medianas (cálculo "
                       "próprio): a cada mês, mediana do crescimento das SPEs do conjunto que têm 12 meses completos (a/a) ou "
                       "dado no início do prazo (base zero), sem contar esta SPE; o número de SPEs aparece ao passar o mouse e "
                       "muda ao longo do tempo (contratos que entram e saem da base). Na tabela, 'mediana dos acumulados' = "
                       "mediana do crescimento p0 → p1 das SPEs com dado nas duas pontas. PIB real: IBGE (SIDRA 1620, índice "
                       "de volume), soma de 4 trimestres. Entrada de praças novas infla o crescimento — use 'Decompor por praça'.")

            # decomposição do crescimento por praça
            if decomp and n_pracas >= 2:
                lpj = lp.reindex(gj.mes)
                if aa:
                    lpa = lp.shift(12).reindex(gj.mes)
                    contrib = (lpj - lpa.fillna(0)).div(lpa.sum(axis=1, min_count=1), axis=0)
                else:
                    contrib = (lpj - lpj.iloc[0]).div(lpj.iloc[0].sum())
                contrib = contrib.loc[:, contrib.abs().sum() > 0]
                if contrib.empty:
                    st.info("Sem variação por praça no prazo escolhido.")
                else:
                    cg = agrupar_pracas(contrib, contrib.abs().mean(), 8)
                    fig = go.Figure([go.Bar(x=cg.index, y=cg[c_], name=str(c_), hovertemplate="%{y:+.1%}") for c_ in cg.columns])
                    fig.add_trace(go.Scatter(x=gj.mes, y=gj.cresc.values, name="Crescimento total", mode="lines",
                                             line=dict(color="black", width=2.5)))
                    fig.update_layout(barmode="relative", bargap=0.1)
                    fig.add_hline(y=0, line_dash="dot", line_color="gray")
                    d1, d2 = st.columns([2.4, 1.1])
                    d1.plotly_chart(layout(fig, "Contribuição de cada praça ao crescimento — " + (
                        "a/a (pontos sobre o LTM de 12 meses antes)" if aa else "acumulado desde o início do prazo"),
                                           h=420, pct=True), width="stretch")
                    b0, b1 = lpj.iloc[0], lpj.iloc[-1]
                    tb = pd.DataFrame({"Praça": lpj.columns, f"LTM {p0}": b0.values, f"LTM {p1}": b1.values})
                    tb["Variação"] = np.where(b0.values > 0, b1.values / np.where(b0.values > 0, b0.values, 1) - 1, np.nan)
                    tb["Contribuição (p.p. do total)"] = (b1.values - b0.values) / b0.sum() if b0.sum() > 0 else np.nan
                    tb["Situação"] = np.where(b0.values <= 0, "nova no prazo", np.where(b1.values <= 0, "saiu no prazo", ""))
                    tb = tb[(tb[f"LTM {p0}"] > 0) | (tb[f"LTM {p1}"] > 0)]
                    tb = tb.reindex(tb["Contribuição (p.p. do total)"].abs().sort_values(ascending=False).index)
                    d2.markdown(f"**{p0} → {p1}, por praça**")
                    d2.dataframe(tb, hide_index=True, width="stretch", height=380,
                                 column_config={f"LTM {p0}": st.column_config.NumberColumn(format="localized"),
                                                f"LTM {p1}": st.column_config.NumberColumn(format="localized"),
                                                "Variação": st.column_config.NumberColumn(format="percent"),
                                                "Contribuição (p.p. do total)": st.column_config.NumberColumn(format="percent")})
                    st.caption("Contribuição (cálculo próprio) = variação do LTM da praça ÷ LTM total da SPE na base (12 meses "
                               "antes no a/a; início do prazo na base zero). A soma das barras dá o crescimento total (linha "
                               "preta). Praça nova entra com contribuição igual a todo o seu volume; se for só renomeação, a "
                               "praça com o nome antigo aparece com contribuição negativa equivalente.")

        pp = t[t.mes > ult - pd.DateOffset(months=12)].groupby("praca", as_index=False)[["veic_eq", "veic_total", "veic_pesado"]].sum()
        pp["pct_pesados"] = pp.veic_pesado / pp.veic_total
        with st.expander("Praças — últimos 12 meses"):
            st.dataframe(pp.sort_values("veic_eq", ascending=False), hide_index=True, width="stretch")

    # ---- financeiro CVM
    st.markdown("### Financeiro (CVM)")
    fo = fin[fin["id"] == op] if not fin.empty else pd.DataFrame()
    if fo.empty:
        if reg is not None and not bool(reg["registro_cvm"]):
            st.warning("SPE sem registro na CVM: sem demonstrações financeiras públicas estruturadas.")
        elif fin.empty:
            st.info("Dados financeiros da CVM ainda não coletados (workflow financeiro-spes no coletor).")
        else:
            st.warning("Sem DFP/ITR estruturada desta SPE na CVM (sem registro ou sem CNPJ casado no cadastro).")
    else:
        per = st.radio("Periodicidade", ["Anual", "LTM (trimestral móvel)"], horizontal=True, key="op_per")
        d = fo[fo.tipo == ("anual" if per == "Anual" else "ltm")].sort_values("dt_fim").copy()
        d["x"] = periodo_rotulo(d)
        if d.empty:
            st.info("Sem períodos nesta periodicidade.")
        else:
            # ---------- painel de comparação: vale para todos os gráficos da seção
            tipo_f = d.tipo.iloc[0]
            base_f = fin[(fin.tipo == tipo_f) & (fin.id != op)].copy()
            base_f["ebitda_aj_por_veq"] = div(base_f.ebitda_aj * 1e6, base_f.veic_eq)
            d["ebitda_aj_por_veq"] = div(d.ebitda_aj * 1e6, d.veic_eq)
            ids_f = sorted(set(base_f.id), key=lambda i: nomes.get(i, i))
            ids_f_gr = [] if info["grupo"] == SEM_GRUPO else [i for i in ids_f if cad.set_index("id").grupo.get(i) == info["grupo"]]
            ids_f_rg = [i for i in ids_f if cad.set_index("id").regiao.get(i) == info["regiao"]]
            st.markdown("**Comparação** — vale para todos os gráficos abaixo")
            f1, f2 = st.columns([1.3, 1.7])
            op_med_f = {f"Grupo {info['grupo']} (demais SPEs, {len(ids_f_gr)})": ("Grupo " + str(info["grupo"]), ids_f_gr, "#ff7f0e"),
                        f"Região {REGIOES.get(info['regiao'], info['regiao'])} (demais SPEs, {len(ids_f_rg)})":
                            ("Região " + REGIOES.get(info["regiao"], str(info["regiao"])), ids_f_rg, "#17becf"),
                        f"Todas as SPEs com CVM (demais, {len(ids_f)})": ("Todas as SPEs", ids_f, "#7f7f7f")}
            op_med_f = {k_: v_ for k_, v_ in op_med_f.items() if v_[1]}
            meds_f = f1.multiselect("Comparar com a mediana de", list(op_med_f), key=f"op_fin_med_{op}",
                                    default=[k_ for k_ in op_med_f if not k_.startswith("Região")])
            cesta_f = f2.multiselect("Cesta de SPEs", ids_f, key=f"op_fin_cesta_{op}",
                                     format_func=lambda i: f"{nomes.get(i, i)} ({cad.set_index('id').grupo.get(i, '')})")
            cesta_modo_f = f2.radio("Mostrar a cesta como", ["Mediana da cesta", "Cada SPE"], horizontal=True,
                                    key="op_fin_cesta_modo", disabled=not cesta_f)
            conj_f = [(f"Mediana — {op_med_f[k_][0]}", op_med_f[k_][1], op_med_f[k_][2], "dash") for k_ in meds_f]
            if cesta_f:
                if cesta_modo_f == "Mediana da cesta":
                    conj_f.append((f"Mediana — cesta ({len(cesta_f)} SPEs)", cesta_f, "#d62728", "dashdot"))
                else:
                    conj_f += [(nomes.get(i_, i_), [i_], PALETA[(j_ + 3) % len(PALETA)], "dot") for j_, i_ in enumerate(cesta_f)]
            st.caption("Medianas (cálculo próprio) das SPEs com DFP/ITR na CVM no mesmo período e periodicidade, sem contar esta "
                       "SPE; o número de SPEs em cada ponto aparece ao passar o mouse. Gráficos em R$ mm não recebem mediana "
                       "(SPEs de tamanhos muito diferentes): a comparação deles é feita pelas versões normalizadas — margens, "
                       "por veículo equivalente, por km, conversão em caixa e % da dívida no curto prazo.")

            def add_comp(fig_, col, pct_=False, yaxis=None, sufixo=""):
                """Acrescenta ao gráfico a mediana de `col` em cada conjunto escolhido no painel (ou cada SPE da cesta)."""
                fmt_ = ".1%" if pct_ else ",.2f"
                for nome_, ids_, cor_, traco_ in conj_f:
                    gb_ = base_f[base_f.id.isin(ids_)].groupby("dt_fim")[col]
                    y_ = d.dt_fim.map(gb_.median())
                    n_ = d.dt_fim.map(gb_.count()).fillna(0).astype(int)
                    if y_.notna().any():
                        fig_.add_trace(go.Scatter(x=d.x, y=y_, name=nome_ + sufixo, mode="lines+markers", yaxis=yaxis,
                                                  line=dict(color=cor_, dash=traco_, width=2), marker=dict(size=5),
                                                  customdata=n_, hovertemplate=f"%{{y:{fmt_}}} · %{{customdata}} SPE(s)"))

            ROT_F = {**ROT, "margem_ebitda_aj": "Margem EBITDA aj.", "margem_ebitda_aj_pos_arrend": "Margem EBITDA aj. pós-arrend.",
                     "ebitda_aj_por_veq": "EBITDA aj. por veíc. eq. (R$)"}
            k1, k2 = st.columns(2)
            fig = go.Figure([go.Bar(x=d.x, y=d.receita_aj, name="Receita líquida ajustada"),
                             go.Bar(x=d.x, y=d.ebitda_aj, name="EBITDA ajustado"),
                             go.Bar(x=d.x, y=d.ebitda_aj_pos_arrend, name="EBITDA aj. pós-arrendamento")])
            k1.plotly_chart(layout(fig, "Receita e EBITDA ajustados (R$ mm)", x=True), width="stretch")
            v_mg = k2.selectbox("Margem comparada", ["margem_ebitda_aj", "margem_ebitda_aj_pos_arrend"],
                                format_func=lambda k: ROT_F[k], key="op_fin_cmp_mg")
            fig = go.Figure([go.Scatter(x=d.x, y=d.margem_ebitda_aj, name="Margem EBITDA aj.", mode="lines+markers", line=dict(width=3)),
                             go.Scatter(x=d.x, y=d.margem_ebitda_aj_pos_arrend, name="Margem pós-arrend.", mode="lines+markers", line=dict(width=3))])
            add_comp(fig, v_mg, pct_=True)
            k2.plotly_chart(layout(fig, f"Margens — medianas: {ROT_F[v_mg].lower()}", pct=True, x=True), width="stretch")
            st.caption("Receita líquida ajustada = receita líquida sem receita de construção (ICPC 01; identificada na DRE, "
                       "DVA ou DFC). EBITDA ajustado = EBIT + D&A − margem de construção + constituição da provisão de "
                       "manutenção (ajuste não caixa da DFC; atualização monetária/AVP pura fica de fora por ser financeira). "
                       "Pós-arrendamento = − pagamentos de arrendamento (principal + juros) da DFC.")
            # receita e EBITDA normalizados (comparáveis entre SPEs de tamanhos diferentes)
            n1, n2 = st.columns([1, 2.4])
            v_nm = n1.selectbox("Receita/EBITDA por unidade", ["tarifa_media", "ebitda_aj_por_veq", "ebitda_aj_por_km"],
                                format_func=lambda k: ROT_F[k], key="op_fin_cmp_norm")
            n1.caption("Tarifa média = receita líquida ajustada ÷ veículos equivalentes do mesmo período; EBITDA por veíc. eq. "
                       "idem; por km = extensão do cadastro. Cálculo próprio.")
            fig = go.Figure([go.Bar(x=d.x, y=d[v_nm], name="Esta SPE", marker_color="#1f77b4", opacity=0.85)])
            add_comp(fig, v_nm)
            n2.plotly_chart(layout(fig, f"{ROT_F[v_nm]} — SPE vs. comparação", h=360, x=True), width="stretch")
            # crescimento vs macro
            st.markdown("**Crescimento vs. PIB nominal, IPCA e dólar**")
            g1, g2 = st.columns([1, 2])
            modo_f = g1.radio("Medida", ["a/a", "Base 100"], horizontal=True, key="op_fin_modo")
            vars_c = g1.multiselect("Variáveis", ["receita_aj", "ebitda_aj", "ebitda_aj_pos_arrend"],
                                    default=["receita_aj", "ebitda_aj"], format_func=lambda k: ROT[k], key="op_fin_vars")
            macro_sel = g1.multiselect("Macro", macro_aba.OPCOES_MACRO, default=["PIB nominal", "IPCA"], key="op_fin_macro")
            v_cr = g1.selectbox("Variável comparada com as medianas", ["receita_aj", "ebitda_aj", "ebitda_aj_pos_arrend"],
                                format_func=lambda k: ROT[k], key="op_fin_cmp_cr", disabled=not conj_f)
            xs = list(d.x)
            if len(xs) > 1:
                p0, p1 = g1.select_slider("Prazo analisado", options=xs, value=(xs[max(0, len(xs) - 6)] if per == "Anual" else xs[max(0, len(xs) - 21)], xs[-1]),
                                          key=f"op_fin_prazo_{op}_{per}")
            else:
                p0 = p1 = xs[0]
            i0, i1 = xs.index(p0), xs.index(p1)
            janela = xs[i0:i1 + 1]
            fins = pd.Series(d.dt_fim.to_numpy(), index=d.x.to_numpy())
            fig = go.Figure()
            resumo = []
            anos_j = max((d.dt_fim.iloc[i1] - d.dt_fim.iloc[i0]).days / 365.25, 1e-9)
            for v in vars_c:
                s = pd.Series(d[v].to_numpy(), index=d.x.to_numpy())
                if modo_f == "a/a":           # a/a calculado na série inteira e depois recortado (o 1º período da janela tem variação)
                    y = (s.pct_change() if per == "Anual" else s / s.shift(4) - 1).loc[janela]
                else:
                    sj = s.loc[janela]
                    b0 = sj.first_valid_index()
                    y = 100 * sj / sj[b0] if b0 is not None and sj[b0] > 0 else sj * np.nan
                fig.add_trace(go.Scatter(x=y.index, y=y.values, mode="lines+markers", name=ROT[v]))
                a_, b_ = s.loc[p0], s.loc[p1]
                ok_ = pd.notna(a_) and pd.notna(b_) and a_ > 0 and b_ > 0
                resumo.append({"Série": ROT[v], "Acumulado": b_ / a_ - 1 if ok_ else np.nan,
                               "Ao ano (CAGR)": (b_ / a_) ** (1 / anos_j) - 1 if ok_ and i1 > i0 else np.nan})
            if macro_sel:
                try:
                    niv = macro_aba.niveis_macro(fins, 12, False, pasta_m)
                    if modo_f == "a/a":
                        ant = macro_aba.niveis_macro(fins - pd.DateOffset(years=1), 12, False, pasta_m)
                    for nome in macro_sel:
                        if modo_f == "a/a":
                            y = (niv[nome] / ant[nome] - 1).loc[janela]
                        else:
                            nj = niv[nome].loc[janela]
                            b0 = nj.first_valid_index()
                            y = 100 * nj / nj[b0] if b0 is not None else nj
                        fig.add_trace(macro_aba.go_scatter(list(y.index), y.values, nome))
                        a_, b_ = niv[nome].loc[p0], niv[nome].loc[p1]
                        ok_ = pd.notna(a_) and pd.notna(b_) and a_ > 0
                        resumo.append({"Série": nome, "Acumulado": b_ / a_ - 1 if ok_ else np.nan,
                                       "Ao ano (CAGR)": (b_ / a_) ** (1 / anos_j) - 1 if ok_ and i1 > i0 else np.nan})
                except Exception as e:  # noqa: BLE001
                    st.caption(f"Macro indisponível: {e}")
            # medianas do crescimento: crescimento de cada SPE primeiro, mediana depois
            if conj_f:
                bv = base_f[["id", "dt_fim", v_cr]].rename(columns={v_cr: "v"})
                dts_j = [fins[x_] for x_ in janela]
                if modo_f == "a/a":
                    ant_ = bv.assign(dt_fim=bv.dt_fim + pd.DateOffset(years=1)).rename(columns={"v": "v0"})
                    cr_ = bv.merge(ant_, on=["id", "dt_fim"], how="inner")
                    cr_["y"] = np.where(cr_.v0 > 0, cr_.v / cr_.v0.where(cr_.v0 > 0) - 1, np.nan)
                else:
                    b0_ = bv[bv.dt_fim == fins[p0]].set_index("id").v
                    cr_ = bv.assign(v0=bv.id.map(b0_))
                    cr_["y"] = np.where(cr_.v0 > 0, 100 * cr_.v / cr_.v0.where(cr_.v0 > 0), np.nan)
                cr_ = cr_[cr_.dt_fim.isin(dts_j)]
                v_p0 = bv[bv.dt_fim == fins[p0]].set_index("id").v
                v_p1 = bv[bv.dt_fim == fins[p1]].set_index("id").v
                ac_ = (v_p1 / v_p0.where(v_p0 > 0) - 1).where(v_p1 > 0).dropna()
                rotulo_v = ROT[v_cr].split(" (")[0]
                for nome_, ids_, cor_, traco_ in conj_f:
                    gb_ = cr_[cr_.id.isin(ids_)].groupby("dt_fim").y
                    y_ = pd.Series([gb_.median().get(t_, np.nan) for t_ in dts_j], index=janela)
                    n_ = [int(gb_.count().get(t_, 0)) for t_ in dts_j]
                    if y_.notna().any():
                        fig.add_trace(go.Scatter(x=janela, y=y_.values, mode="lines+markers", name=f"{nome_} ({rotulo_v})",
                                                 line=dict(color=cor_, dash=traco_, width=2), marker=dict(size=5), customdata=n_,
                                                 hovertemplate=("%{y:.1%}" if modo_f == "a/a" else "%{y:.1f}") + " · %{customdata} SPE(s)"))
                    a_ids = ac_[ac_.index.isin(ids_)]
                    if len(a_ids) and i1 > i0:
                        med_ = a_ids.median()
                        resumo.append({"Série": f"{nome_} ({rotulo_v})" + (f" · {len(a_ids)} SPEs" if len(ids_) > 1 else ""),
                                       "Acumulado": med_, "Ao ano (CAGR)": (1 + med_) ** (1 / anos_j) - 1 if med_ > -1 else np.nan})
            if modo_f == "a/a":
                fig.add_hline(y=0, line_dash="dot", line_color="gray")
            g2.plotly_chart(layout(fig, f"Crescimento {p0} → {p1} — " + ("contra o mesmo período do ano anterior" if modo_f == "a/a" else "acumulado (base 100 no início)"),
                                   pct=modo_f == "a/a", x=True), width="stretch")
            if resumo and i1 > i0:
                g1.markdown(f"**{p0} → {p1}** ({anos_j:.1f} anos)")
                g1.dataframe(pd.DataFrame(resumo), hide_index=True, width="stretch",
                             column_config={"Série": st.column_config.TextColumn(width="large"),
                                            "Acumulado": st.column_config.NumberColumn(format="percent"),
                                            "Ao ano (CAGR)": st.column_config.NumberColumn(format="percent")})
            g2.caption("Escolha o início e o fim na barra 'Prazo analisado'. Acumulado e CAGR comparam o nível do fim com o do "
                       "início (cálculo próprio). Medianas: crescimento de cada SPE primeiro (a/a contra o mesmo período do ano "
                       "anterior; base 100 no início do prazo), mediana depois; só entram SPEs com valor positivo na base. Na "
                       "tabela, mediana dos acumulados das SPEs com dado positivo nas duas pontas. Macro no mesmo recorte (12 meses até a data de fim): IPCA e R\\$/US\\$ pela média "
                       "do período; PIB nominal = volume (IBGE) × nível médio de preços (IPCA). R\\$/US\\$ positivo = real se "
                       "desvalorizou. Fonte macro: IBGE e BCB via Price Discovery Scanning.")

            # conversão do EBITDA ajustado em FCO
            st.markdown("**Conversão do EBITDA ajustado em caixa (FCO)**")
            k1, k2 = st.columns(2)
            fig = go.Figure([go.Bar(x=d.x, y=d.ebitda_aj, name="EBITDA ajustado", opacity=0.5),
                             go.Bar(x=d.x, y=d.ebitda_aj_pos_arrend, name="EBITDA aj. pós-arrendamento", opacity=0.75),
                             go.Bar(x=d.x, y=d.fco_comp_pos_arrend, name="FCO comparável pós-arrendamento", opacity=0.95)])
            k1.plotly_chart(layout(fig, "EBITDA ajustado e FCO (R$ mm)", h=380, x=True), width="stretch")
            fig = go.Figure([go.Scatter(x=d.x, y=d.conversao_fco, name="Esta SPE", mode="lines+markers", line=dict(width=3))])
            add_comp(fig, "conversao_fco", pct_=True)
            fig.add_hline(y=1, line_dash="dot", line_color="gray")
            k2.plotly_chart(layout(fig, "Conversão: FCO comp. pós-arrend. ÷ EBITDA aj. pós-arrend.", h=380, pct=True, x=True),
                            width="stretch")
            st.caption("Mesma lógica do Price Discovery Scanning, partindo do EBITDA ajustado de rodovia. FCO comparável = FCO "
                       "reportado sem juros e dividendos pagos/recebidos que a empresa classifica no operacional; aqui também "
                       "sem os pagamentos de arrendamento (principal + juros), para ficar na mesma base do EBITDA ajustado "
                       "pós-arrendamento. Abaixo de 100%: parte do EBITDA ficou em capital de giro, IR/CS pagos e no pagamento "
                       "da provisão de manutenção (que está no FCO, mas foi somada de volta ao EBITDA ajustado). Vazio com "
                       "EBITDA ≤ 0. Cálculo próprio.")

            # prazos, ciclo e perfil da dívida
            st.markdown("**Prazos, ciclo financeiro e perfil de vencimentos**")
            k1, k2 = st.columns(2)
            v_pz = k1.selectbox("Prazo comparado", ["ciclo_financeiro_dias", "pmr_dias", "pmp_dias"],
                                format_func=lambda k: ROT[k], key="op_fin_cmp_pz")
            fig = go.Figure()
            for c_, cor_ in (("pmr_dias", "#1f77b4"), ("pmp_dias", "#9467bd"), ("ciclo_financeiro_dias", "#2ca02c")):
                fig.add_trace(go.Scatter(x=d.x, y=d[c_], name=ROT[c_], mode="lines+markers", line=dict(color=cor_, width=3)))
            add_comp(fig, v_pz, sufixo=f" ({ROT[v_pz].split(' (')[0]})")
            k1.plotly_chart(layout(fig, f"Prazos médios e ciclo financeiro (dias) — medianas: {ROT[v_pz].split(' (')[0]}", h=400, x=True),
                            width="stretch")
            k2.markdown("&nbsp;")
            k2.caption("Linha (eixo da direita): % da dívida no curto prazo, com as medianas da comparação.")
            fig = go.Figure([go.Bar(x=d.x, y=d.divida_cp, name="Curto prazo (BP 2.01.04)"),
                             go.Bar(x=d.x, y=d.divida_lp, name="Longo prazo (BP 2.02.01)"),
                             go.Scatter(x=d.x, y=d.pct_divida_cp, name="% no curto prazo (esta SPE)", yaxis="y2", mode="lines+markers",
                                        line=dict(color="#d4a017", width=3))])
            add_comp(fig, "pct_divida_cp", pct_=True, yaxis="y2", sufixo=" (% CP)")
            fig.update_layout(barmode="stack", yaxis2=dict(overlaying="y", side="right", tickformat=".0%", showgrid=False,
                                                           rangemode="tozero"))
            k2.plotly_chart(layout(fig, "Empréstimos e financiamentos: curto x longo prazo (R$ mm) e % no curto prazo", h=400, x=True),
                            width="stretch")
            st.caption("PMR = clientes ÷ receita líquida ajustada × dias; PMP = fornecedores ÷ custo (sem construção) × dias; "
                       "ciclo financeiro = PMR − PMP (concessão não tem estoque relevante). Tracejados: medianas da comparação "
                       "escolhida no painel, para o prazo selecionado. Cálculo próprio sobre a CVM.")
            deb_o = deb[deb["id"] == op] if not deb.empty else pd.DataFrame()
            if len(deb_o):
                cdo = custo_debentures(deb_o, pj)
                cdo["ano_venc"] = cdo.vencimento.dt.year
                k1, k2 = st.columns([1, 1.4])
                v = cdo.groupby("ano_venc", as_index=False).saldo_rs_mm.sum()
                k1.plotly_chart(layout(px.bar(v, x="ano_venc", y="saldo_rs_mm", labels={"ano_venc": "", "saldo_rs_mm": "R$ mm"}),
                                       "Debêntures: saldo por ano de vencimento final", x=True), width="stretch")
                k2.dataframe(cdo[["codigo", "remuneracao", "vencimento", "saldo_rs_mm", "prazo_anos", "taxa_contratada",
                                  "taxa_mercado", "spread_emissao", "spread_mercado", "nominal_emissao", "nominal_mercado"]]
                             .sort_values("vencimento").round({c_: 2 for c_ in ("saldo_rs_mm", "prazo_anos", "taxa_contratada", "taxa_mercado", "spread_emissao", "spread_mercado", "nominal_emissao", "nominal_mercado")}), hide_index=True, width="stretch")
                k2.caption(f"Fonte: {fonte_deb}. Vencimento final (sem a curva de amortização da escritura). Spreads em p.p. "
                           "sobre o soberano no prazo (CDI+: spread sobre o CDI).")
                # comparação: esta SPE x demais SPEs do grupo x universo de concessionárias de rodovias
                cd_all = custo_debentures(deb.drop_duplicates("codigo"), pj)
                cd_all["grupo"] = cd_all["id"].map(cad.set_index("id").grupo)
                cods = set(deb_o.codigo)
                # conjuntos do painel de comparação (sem esta SPE); aqui ponderado pelo saldo, não mediana
                ids_deb = {}
                for k_ in meds_f:
                    rot_c, _ids, _cor = op_med_f[k_]
                    ids_deb[rot_c] = (set(cad[cad.grupo == info["grupo"]].id) if rot_c.startswith("Grupo")
                                      else set(cad[cad.regiao == info["regiao"]].id) if rot_c.startswith("Região")
                                      else set(cad.id), _cor)
                if cesta_f:
                    if cesta_modo_f == "Mediana da cesta":
                        ids_deb[f"Cesta ({len(cesta_f)} SPEs)"] = (set(cesta_f), "#d62728")
                    else:
                        for j_, i_ in enumerate(cesta_f):
                            ids_deb[nomes.get(i_, i_)] = ({i_}, PALETA[(j_ + 3) % len(PALETA)])
                conjuntos = {"Esta SPE": (cd_all[cd_all.codigo.isin(cods)], "#1f77b4")}
                for rot_c, (ids_c, cor_c) in ids_deb.items():
                    conjuntos[rot_c] = (cd_all[cd_all["id"].isin(ids_c) & ~cd_all.codigo.isin(cods)], cor_c)
                tabs_c, cores_c = [], []
                for nome_c, (dfc, cor_c) in conjuntos.items():
                    if len(dfc):
                        r_c = resumo_custo(dfc)
                        r_c.insert(0, "Conjunto", nome_c + (f" · {dfc['id'].nunique()} SPEs" if nome_c != "Esta SPE" else ""))
                        tabs_c.append(r_c)
                        cores_c.append(cor_c)
                if tabs_c:
                    comp_d = pd.concat(tabs_c, ignore_index=True)
                    st.markdown("**Remuneração da dívida: esta SPE vs. conjuntos do painel de comparação** (debêntures, ponderado pelo saldo)")
                    q1, q2 = st.columns([1, 1.6])
                    med_d = q1.radio("Medida", ["Spread mercado s/ soberano (p.p.)", "Spread emissão s/ soberano (p.p.)",
                                                "Custo nominal mercado (% a.a.)", "Custo nominal emissão (% a.a.)",
                                                "Spread nominal mercado s/ pré (p.p.)", "Spread nominal emissão s/ pré (p.p.)"],
                                     key="op_deb_med")
                    g_d = comp_d[comp_d.Indexador != "Outros"].copy()
                    g_d["Indexador"] = g_d.Indexador.str.replace(" (nominal equivalente)", "", regex=False)
                    fig = px.bar(g_d, x="Indexador", y=med_d, color="Conjunto", barmode="group", text_auto=".2f",
                                 color_discrete_sequence=cores_c,
                                 labels={"Indexador": "", med_d: ""})
                    q2.plotly_chart(layout(fig, med_d, h=380), width="stretch")
                    q1.caption("Spreads por indexador: CDI+ = spread sobre o CDI; IPCA+ e prefixada = taxa − ETTJ ANBIMA no "
                               "mesmo prazo. 'Total' usa o custo nominal equivalente (todas as classes na curva de hoje). "
                               f"Conjuntos = debêntures das demais SPEs de cada conjunto do painel (base: {cd_all.codigo.nunique()} "
                               "papéis de SPEs com CNPJ); como as taxas dependem do saldo, aqui vale a média ponderada pelo saldo, "
                               "não a mediana. Cálculo próprio.")
                    with st.expander("Tabela da comparação"):
                        st.dataframe(comp_d.round(2), hide_index=True, width="stretch")

            # capex
            st.markdown("**Capex e capex de manutenção**")
            k1, k2 = st.columns(2)
            fig = go.Figure([go.Bar(x=d.x, y=d.capex, name="Capex"), go.Bar(x=d.x, y=d.capex_manut, name="Capex de manutenção")])
            k1.plotly_chart(layout(fig, "Capex (R$ mm)", x=True), width="stretch")
            aprox = d[d.capex_manut_aprox.astype(bool)].x.tolist()
            k1.caption("Capex = aquisições de imobilizado, intangível e ativo de contrato (DFC). Capex de manutenção = "
                       "pagamento/utilização da provisão de manutenção na DFC."
                       + (f" **Aproximado** (pagamento não identificado na DFC; saldo inicial + ajuste da DFC − saldo final) em: "
                          f"{', '.join(aprox)}." if aprox else ""))
            # benchmark capex por VEQ e por km — todos os períodos
            metr_b = k2.selectbox("Indicador de capex", ["capex_por_veq", "capex_manut_por_veq", "capex_por_km", "capex_manut_por_km"],
                                  format_func=lambda k: ROT[k], key="op_bench")
            fig = go.Figure([go.Bar(x=d.x, y=d[metr_b], name="Esta SPE", marker_color="#1f77b4", opacity=0.85)])
            add_comp(fig, metr_b)
            k2.plotly_chart(layout(fig, f"{ROT[metr_b]} — SPE vs. comparação", h=380, x=True), width="stretch")
            k2.caption("Medianas/cesta do painel de comparação. Por veículo equivalente usa o tráfego do mesmo período; por "
                       "km, a extensão do cadastro. Cálculo próprio.")

            # ICSD e alavancagem
            st.markdown("**ICSD e alavancagem**")
            k1, k2 = st.columns(2)
            v_ic = k1.selectbox("ICSD comparado", ["icsd", "icsd_sem_manut", "icsd_arr", "icsd_arr_sem_manut"],
                                format_func=lambda k: ROT[k], key="op_fin_cmp_icsd")
            v_dl = k2.selectbox("Alavancagem comparada", ["dl_ebitda_aj_ifrs16", "dl_ebitda_aj_pre_ifrs16"],
                                format_func=lambda k: ROT[k], key="op_fin_cmp_dl")
            AZ, LJ = "#2E6FBA", "#E69F00"
            fig = go.Figure([
                go.Scatter(x=d.x, y=d.icsd, mode="lines+markers", name="(1) Regra geral — com capex de manutenção",
                           line=dict(width=3, color=AZ)),
                go.Scatter(x=d.x, y=d.icsd_sem_manut, mode="lines+markers", name="(1) Regra geral — sem pagamento da provisão",
                           line=dict(dash="dash", color=AZ)),
                go.Scatter(x=d.x, y=d.icsd_arr, mode="lines+markers", name="(2) Arrend. no denominador — com capex de manutenção",
                           line=dict(width=3, color=LJ)),
                go.Scatter(x=d.x, y=d.icsd_arr_sem_manut, mode="lines+markers", name="(2) Arrend. no denominador — sem pagamento da provisão",
                           line=dict(dash="dash", color=LJ))])
            for nivel, corl in ((1.0, "#C0504D"), (1.2, "#8C8C8C"), (1.3, "#4D4D4D")):
                fig.add_hline(y=nivel, line_dash="dot", line_color=corl, annotation_text=f"ICSD {nivel:.1f}x".replace(".", ","), annotation_position="top left", annotation_font_size=11)
            mx_ = d[d.icsd_base_mista.fillna(False).astype(bool) & d.icsd_arr.notna()]
            if len(mx_):
                fig.add_trace(go.Scatter(x=mx_.x, y=mx_.icsd_arr, mode="markers", name="(2) com linha mista dívida + arrendamento",
                                         marker=dict(symbol="diamond-open", size=9, color="#C0504D", line=dict(width=1.5))))
            add_comp(fig, v_ic, sufixo=" (" + ROT[v_ic].split(",")[0].replace("ICSD ", "") + ")")
            k1.plotly_chart(layout(fig, "ICSD (x)", x=True), width="stretch")
            k1.caption("(1) Regra geral = (EBITDA aj. − arrendamento − IR/CS pagos − capex de manutenção) ÷ (juros + principal da "
                       "dívida). (2) = (EBITDA aj. − IR/CS pagos − capex de manutenção) ÷ (juros + principal + arrendamento). "
                       "Com ICSD acima de 1x, (2) fica abaixo de (1); a distância cresce com o peso do arrendamento. Tracejado: "
                       "ignora o pagamento da provisão de manutenção. Cálculo próprio com dados da DFC (CVM).")
            if len(mx_):
                k1.caption("◇ Períodos em que a DFC paga dívida e arrendamento numa única linha (" + ", ".join(mx_.x.astype(str)) +
                           "): só a fórmula (2) é calculável, com a linha mista no denominador.")
                baixo_ = mx_[(mx_.servico_icsd_arr / mx_.divida_bruta) < 0.04]
                if len(baixo_):
                    k1.warning("Atenção em " + ", ".join(baixo_.x.astype(str)) + ": o serviço considerado é menor que 4% da "
                               "dívida bruta, abaixo do que só os juros custariam. Os juros provavelmente estão capitalizados ou "
                               "em outra linha, e o ICSD desses períodos tende a estar superestimado (avaliação própria).")
            sem_icsd = d[d.icsd.isna()]
            if len(sem_icsd):
                mot = [(motivo_sem_icsd(r), x) for x, r in zip(sem_icsd.x, sem_icsd.to_dict("records"))]
                partes = []
                for m_ in dict.fromkeys(m for (m, _), _x in mot):
                    det = ", ".join(f"{x}" + (f" [{dt}]" if dt else "") for (mm, dt), x in mot if mm == m_)
                    partes.append(f"**{m_}** — {det}")
                todos = d.icsd.isna().all() and d.icsd_arr.isna().all()
                (k1.warning if todos else k1.caption)(
                    ("ICSD vazio em todos os períodos. " if todos else "Períodos sem ICSD (1) regra geral: ") + "; ".join(partes) +
                    ". (Diagnóstico próprio a partir das linhas da DFC/BP na CVM.)")
            fig = go.Figure([go.Scatter(x=d.x, y=d.dl_ebitda_aj_ifrs16, mode="lines+markers", name="DL c/ arrend. ÷ EBITDA aj. (IFRS 16)"),
                             go.Scatter(x=d.x, y=d.dl_ebitda_aj_pre_ifrs16, mode="lines+markers", name="DL s/ arrend. ÷ EBITDA aj. pós-arrend. (pré-IFRS 16)")])
            add_comp(fig, v_dl, sufixo=" (" + ("IFRS 16" if v_dl.endswith("ifrs16") and "pre" not in v_dl else "pré-IFRS 16") + ")")
            k2.plotly_chart(layout(fig, "Dívida líquida ÷ EBITDA ajustado (x)", x=True), width="stretch")
            st.caption("Alavancagem como no Price "
                       "Discovery Scanning: com arrendamento (IFRS 16) contra EBITDA ajustado; sem arrendamento contra EBITDA "
                       "ajustado pós-arrendamento.")
            u_ = d.assign(_ok=d.icsd.notna() | d.icsd_arr.notna()).query("_ok").dropna(subset=["veic_eq", "receita_aj"]).tail(1)
            if len(u_):
                r_ = u_.iloc[0]
                tar = r_.receita_aj / r_.veic_eq
                be = []
                for alvo_ in (1.0, 1.2, 1.3):
                    lin = {"ICSD alvo": f"{alvo_:.1f}x"}
                    for rot_, cf_, sv_ in (("(1)", "cfads", "servico_icsd"), ("(2)", "cfads_arr", "servico_icsd_arr")):
                        if pd.notna(r_[sv_]) and r_[sv_] > 0:
                            veq_be = r_.veic_eq - (r_[cf_] - alvo_ * r_[sv_]) / tar
                            lin[f"{rot_} Veíc. eq. de equilíbrio (mi)"] = veq_be / 1e6
                            lin[f"{rot_} Variação tolerada do tráfego"] = veq_be / r_.veic_eq - 1
                    be.append(lin)
                st.markdown(f"**Tráfego de equilíbrio — {r_.x}** (hipótese própria: receita proporcional ao tráfego, "
                            f"custos, IR, arrendamento e capex de manutenção fixos; tráfego atual {r_.veic_eq / 1e6:,.1f} mi veíc. eq.)")
                st.dataframe(pd.DataFrame(be), hide_index=True, width="content",
                             column_config={**{f"{k_} Veíc. eq. de equilíbrio (mi)": st.column_config.NumberColumn(format="%.1f") for k_ in ("(1)", "(2)")},
                                            **{f"{k_} Variação tolerada do tráfego": st.column_config.NumberColumn(format="percent") for k_ in ("(1)", "(2)")}})
                st.caption("IR fixo é conservador (com menos receita o IR cairia). Variação negativa = queda de tráfego que a "
                           "SPE suporta até o ICSD alvo; positiva = já está abaixo do alvo.")

            # estrutura societária (esquema) e peso no grupo
            st.markdown("**Estrutura societária — quem investe em quem**")
            ac = ler_acionistas_curadoria()
            ac = ac[ac.id == op] if not ac.empty else ac
            fre = ler_fin("acionistas_fre")
            if not fre.empty:
                fre = fre.assign(id=fre["ids"].astype(str).str.split("|")).explode("id")
                fre = fre[fre.id == op]
            bruto_fre = ler_fin("acionistas_fre_bruto")
            e1, e2 = st.columns(2)
            niv_e = e1.slider("Níveis acima da SPE", 1, 6, 3, key="op_niveis")
            pmin_e = e2.slider("Participação mínima exibida (%)", 0.0, 50.0, 5.0, 0.5, key="op_pmin",
                               help="Vale do 2º nível para cima; os acionistas diretos da SPE aparecem sempre.")
            dot, nota_dot = esquema_societario(op, info, ac, fre, bruto_fre, niv_e, pmin_e)
            if dot:
                st.graphviz_chart(dot, width="stretch")
                st.caption(nota_dot)
            else:
                st.caption("Sem estrutura societária na base (FRE não entregue/coletado e sem curadoria).")
            if len(ac) or len(fre):
                with st.expander("Tabelas de origem (curadoria e Formulário de Referência)"):
                    if len(ac):
                        st.dataframe(ac.drop(columns="id"), hide_index=True, width="stretch")
                    if len(fre):
                        st.dataframe(fre[["acionista", "participacao_pct", "controlador", "data_referencia"]], hide_index=True,
                                     width="stretch")

            ua = fo[fo.tipo == "anual"]
            if len(ua) and info["grupo"] != SEM_GRUPO:
                st.markdown(f"**Peso da SPE no grupo {info['grupo']}**")
                vars_g = ["receita_aj", "ebitda_aj", "capex", "capex_manut", "divida_bruta", "veic_eq"]
                ga = fin[(fin.tipo == "anual") & (fin.grupo == info["grupo"])]
                ex = int(ua.exercicio.max())
                gr = ga[ga.exercicio == ex]
                k1, k2 = st.columns(2)
                tot = gr[vars_g].sum(min_count=1)
                eu = ua[ua.exercicio == ex][vars_g].iloc[0]
                pesos = pd.DataFrame({"Variável": [ROT[k_].split(" (")[0] for k_ in vars_g], "pct": (eu / tot).values,
                                      "SPE": eu.values, "Grupo": tot.values})
                fig = px.bar(pesos, x="pct", y="Variável", orientation="h", text_auto=".0%",
                             color_discrete_sequence=[COR_GRUPO.get(info["grupo"], "#1f77b4")],
                             hover_data={"SPE": ":.1f", "Grupo": ":.1f", "pct": ":.1%"}, labels={"pct": "", "Variável": ""})
                fig.update_xaxes(tickformat=".0%", range=[0, 1])
                k1.plotly_chart(layout(fig, f"% do grupo — exercício {ex} ({gr.id.nunique()} SPEs do grupo com DFP)", h=360),
                                width="stretch")
                comp_g = gr.melt(id_vars=["nome"], value_vars=vars_g, var_name="var", value_name="valor").dropna()
                comp_g["Variável"] = comp_g["var"].map(lambda k_: ROT[k_].split(" (")[0])
                cores_spe = {n_: ("#d62728" if n_ == info["nome"] else c_) for n_, c_ in
                             zip(sorted(gr.nome.unique()), px.colors.qualitative.Pastel * 5)}
                fig = px.bar(comp_g, x="valor", y="Variável", color="nome", orientation="h", color_discrete_map=cores_spe,
                             labels={"valor": "", "Variável": "", "nome": "SPE"})
                fig.update_layout(barnorm="percent", barmode="stack")
                fig.update_xaxes(ticksuffix="%")
                k2.plotly_chart(layout(fig, f"Composição do grupo por SPE — {ex} (em vermelho: {info['nome']})", h=360),
                                width="stretch")
                var_t = st.selectbox("Peso no grupo ao longo do tempo", vars_g, format_func=lambda k_: ROT[k_], key="op_peso_t")
                tg = ga.groupby("exercicio")[var_t].sum(min_count=1)
                ts = ua.set_index("exercicio")[var_t]
                serie_p = (ts / tg.reindex(ts.index)).dropna()
                fig = go.Figure([go.Bar(x=serie_p.index.astype(str), y=serie_p.values, marker_color=COR_GRUPO.get(info["grupo"], "#1f77b4"),
                                        text=[f"{v:.0%}" for v in serie_p.values], textposition="outside")])
                fig.update_yaxes(tickformat=".0%")
                st.plotly_chart(layout(fig, f"{ROT[var_t]}: participação da SPE no grupo por exercício", h=320, x=True),
                                width="stretch")
                st.caption("Grupo = soma das SPEs do grupo com DFP no exercício (o número de SPEs muda ao longo do tempo com "
                           "entradas e saídas de concessões). Não é o consolidado da holding. Cálculo próprio.")
            with st.expander("Tabela financeira completa"):
                cols_f = ["x", "fonte", "receita_aj", "receita_construcao", "ebitda", "prov_manut_constituicao", "ebitda_aj",
                          "pagamento_arrendamento", "ebitda_aj_pos_arrend", "ir_csll_pagos", "capex", "capex_manut",
                          "capex_manut_aprox", "prov_manut_saldo", "cfads", "juros_pagos", "principal_pago",
                          "servico_divida_pago", "servico_div_arrend_misto", "servico_icsd_arr", "icsd_base_mista", "icsd", "icsd_sem_manut", "icsd_arr", "icsd_arr_sem_manut", "divida_bruta", "caixa", "divida_liquida",
                          "passivo_arrendamento", "dl_ebitda_aj_ifrs16", "dl_ebitda_aj_pre_ifrs16", "custo_implicito",
                          "pmr_dias", "pmp_dias", "ciclo_financeiro_dias", "veic_eq", "tarifa_media", "notas"]
                tab_f = d[[c_ for c_ in cols_f if c_ in d.columns]]
                st.dataframe(tab_f, hide_index=True, width="stretch")
                baixar_botao(tab_f, f"financeiro_{op}_{d.tipo.iloc[0]}")

    # ---- dados econômicos ANTT (reportados ao regulador)
    if not t.empty and t.fonte.iloc[0] == "ANTT":
        rec, cus, inv = anual_antt("receita_pedagio"), anual_antt("custos_operacionais"), anual_antt("investimentos")
        fa = rec[rec.id_concessao == op].rename(columns={"valor": "receita_pedagio"}) if not rec.empty else pd.DataFrame()
        for nome, dd in (("custos_operacionais", cus), ("investimentos", inv)):
            if len(fa) and not dd.empty:
                fa = fa.merge(dd[dd.id_concessao == op].rename(columns={"valor": nome}), on=["id_concessao", "ano"], how="outer")
        if len(fa):
            with st.expander("Dados econômicos reportados à ANTT (R$, anual)"):
                eq_ano = t.groupby(t.mes.dt.year).veic_eq.sum()
                fa["tarifa_media_por_veic_eq"] = fa.receita_pedagio / fa.ano.map(eq_ano)
                st.plotly_chart(px.bar(fa.sort_values("ano"), x="ano", y=[c_ for c_ in ("receita_pedagio", "custos_operacionais", "investimentos") if c_ in fa],
                                       barmode="group", labels={"value": "R$", "ano": ""}), width="stretch")
                st.dataframe(fa.sort_values("ano"), hide_index=True, width="stretch")
    if len(m):
        baixar_botao(m, f"trafego_{op}")

# ================================================================== comparar
with aba_comp:
    st.markdown("Métricas de tráfego calculadas sobre ANTT/ARTESP (LTM = últimos 12 meses com dado da SPE) e "
                "financeiras sobre a CVM (último período escolhido). Cálculo próprio.")
    c1, c2 = st.columns(2)
    chave = c1.selectbox("Grupo de pares (chave de comparação)", ["regiao_nome", "uf_principal", "agencia", "grupo", "(todas)"],
                         format_func=lambda c: {"regiao_nome": "Região", "uf_principal": "UF", "agencia": "Agência",
                                                "grupo": "Grupo controlador", "(todas)": "Todas"}[c])
    per_c = c2.radio("Financeiro", ["Último LTM", "Último exercício"], horizontal=True, key="cmp_per")
    num_t = ["veic_eq_ltm", "cresc_ltm", "cresc_ltm_mesmas_pracas", "pracas_novas_ltm", "cagr_3a", "pct_pesados",
             "vdm_eq_por_km", "pior_queda_anual"]
    comp = sel[["id", "nome", "grupo", "regiao_nome", "uf_principal", "agencia", "ultimo_mes"] + num_t].copy()
    num_f = ["receita_aj", "ebitda_aj", "margem_ebitda_aj", "ebitda_aj_pos_arrend", "capex", "capex_manut", "icsd",
             "icsd_sem_manut", "icsd_arr", "icsd_arr_sem_manut", "dl_ebitda_aj_ifrs16", "dl_ebitda_aj_pre_ifrs16", "divida_bruta", "divida_liquida",
             "custo_implicito", "tarifa_media", "capex_por_veq", "capex_manut_por_veq", "capex_por_km", "capex_manut_por_km",
             "ebitda_aj_por_km", "pmr_dias", "pmp_dias", "ciclo_financeiro_dias", "pct_divida_cp", "fco_comp_pos_arrend",
             "conversao_fco"]
    if not fin.empty:
        tp = "ltm" if per_c == "Último LTM" else "anual"
        uf_ = fin[fin.tipo == tp].sort_values("dt_fim").groupby("id").tail(1).set_index("id")
        uf_["periodo_fin"] = periodo_rotulo(uf_)
        comp["periodo_fin"] = comp.id.map(uf_["periodo_fin"])
        for c_ in num_f:
            comp[c_] = comp.id.map(uf_[c_])
    num = num_t + ([c_ for c_ in num_f if c_ in comp] if not fin.empty else [])

    st.subheader("Ranking por indicador")
    r1, r2, r3 = st.columns([2, 1, 1])
    ind = r1.selectbox("Indicador", num, index=num.index("icsd") if "icsd" in num else 0, format_func=lambda k: ROT.get(k, k))
    ordem = r2.radio("Ordem", ["Maior primeiro", "Menor primeiro"], horizontal=True)
    topn = r3.number_input("Mostrar (0 = todas)", 0, 200, 0)
    rk = comp.dropna(subset=[ind]).sort_values(ind, ascending=ordem == "Menor primeiro")
    if topn:
        rk = rk.head(int(topn))
    rk["posição"] = range(1, len(rk) + 1)
    if len(rk):
        fig = px.bar(rk, x=ind, y="nome", color="grupo", orientation="h", color_discrete_map=COR_GRUPO,
                     hover_data=["regiao_nome", "uf_principal"], labels={ind: ROT.get(ind, ind), "nome": "", "grupo": "Grupo"})
        fig.update_yaxes(categoryorder="array", categoryarray=list(rk.nome)[::-1])
        if ind in PCT:
            fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(layout(fig, f"Ranking — {ROT.get(ind, ind)} ({len(rk)} SPEs com dado)", h=max(360, 22 * len(rk) + 120)),
                        width="stretch")
        med = rk.groupby(chave)[ind].median() if chave != "(todas)" else None
        st.dataframe(rk[["posição", "nome", "grupo", "regiao_nome", "uf_principal", ind] + (["periodo_fin"] if "periodo_fin" in rk else [])],
                     hide_index=True, width="stretch")
        if med is not None:
            st.caption("Mediana por " + {"regiao_nome": "região", "uf_principal": "UF", "agencia": "agência", "grupo": "grupo"}[chave]
                       + ": " + " · ".join(f"{k}: {v:.3g}" for k, v in med.items()))
    else:
        st.info("Nenhuma SPE com dado para este indicador.")

    st.subheader("Tabela e percentis")
    if chave != "(todas)":
        for c_ in num:
            comp[f"pctl_{c_}"] = comp.groupby(chave)[c_].rank(pct=True)
    st.dataframe(comp.drop(columns="id"), hide_index=True, width="stretch")
    baixar_botao(comp.drop(columns="id"), "comparacao_spes")

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("Eixo X", num, index=num.index("cresc_ltm_mesmas_pracas"), format_func=lambda k: ROT.get(k, k))
    y = c2.selectbox("Eixo Y", num, index=num.index("vdm_eq_por_km"), format_func=lambda k: ROT.get(k, k))
    cor_s = c3.selectbox("Cor", ["grupo", "regiao_nome", "agencia", "uf_principal"])
    st.plotly_chart(px.scatter(comp, x=x, y=y, color=cor_s, hover_name="nome", size=comp.veic_eq_ltm.fillna(0).clip(lower=1),
                               color_discrete_map=COR_GRUPO if cor_s == "grupo" else None,
                               labels={x: ROT.get(x, x), y: ROT.get(y, y)}), width="stretch")

    st.markdown("**Tráfego base 100**")
    escolha = st.multiselect("SPEs", comp.id.tolist(), format_func=lambda i: nomes.get(i, i),
                             default=comp.sort_values("veic_eq_ltm", ascending=False).id.head(5).tolist())
    if escolha:
        b = traf[traf.id_concessao.isin(escolha)].groupby(["id_concessao", "mes"], as_index=False).veic_eq.sum()
        b["ltm"] = b.groupby("id_concessao").veic_eq.transform(lambda s: s.rolling(12).sum())
        b = b.dropna(subset=["ltm"])
        if len(b):
            inicio = st.slider("Início da base", min_value=b.mes.min().to_pydatetime(), max_value=b.mes.max().to_pydatetime(),
                               value=max(b.mes.min(), b.mes.max() - pd.DateOffset(years=5)).to_pydatetime(), format="MM/YYYY")
            b = b[b.mes >= inicio]
            b["base100"] = b.groupby("id_concessao").ltm.transform(lambda s: s / s.iloc[0] * 100)
            b["SPE"] = b.id_concessao.map(nomes)
            st.plotly_chart(px.line(b, x="mes", y="base100", color="SPE", labels={"mes": "", "base100": "LTM, base 100"}),
                            width="stretch")
            st.caption("Série LTM (12 meses acumulados) para neutralizar sazonalidade.")

# ================================================================== juros
with aba_jur:
    if pj.empty:
        st.info("Curvas indisponíveis (Price Discovery Scanning, saidas/juros/).")
    else:
        datas_j = sorted(pj["data"].unique())
        st.caption(f"Fonte: ANBIMA, Estrutura a Termo das Taxas de Juros (modelo Svensson), {len(datas_j)} datas de "
                   f"{pd.Timestamp(datas_j[0]).date()} a {pd.Timestamp(datas_j[-1]).date()}, via Price Discovery Scanning. "
                   "Taxas % a.a., base 252. A ANBIMA libera gratuitamente só os últimos 5 dias úteis: o histórico cresce com a coleta diária.")
        tipos_c = {"Prefixada": "pre", "Real (IPCA)": "ipca", "Inflação implícita": "implicita"}
        j1, j2, j3 = st.columns([1.3, 1.2, 2])
        tc_rot = j1.radio("Curva", list(tipos_c), key="jur_curva")
        tc = tipos_c[tc_rot]
        base_j = j2.selectbox("Data", datas_j[::-1], format_func=lambda d: pd.Timestamp(d).strftime("%d/%m/%Y"), key="jur_data")
        prazos_cmp = {"1 semana antes": 7, "1 mês antes": 30, "3 meses antes": 91, "6 meses antes": 182, "1 ano antes": 365}
        sel_cmp = j3.multiselect("Comparar com", list(prazos_cmp), default=["1 mês antes", "1 ano antes"], key="jur_cmp")
        extra = j3.multiselect("Outras datas", [d for d in datas_j[::-1] if d != base_j],
                               format_func=lambda d: pd.Timestamp(d).strftime("%d/%m/%Y"), key="jur_extra")
        max_anos = st.slider("Prazo máximo (anos)", 2, 30, 10, key="jur_max",
                             help="A ANBIMA publica vértices prefixados até 10 anos; além disso a curva é extrapolação do Svensson.")
        anos = np.linspace(0.25, max_anos, 120)
        datas_plot, faltam = [(pd.Timestamp(base_j), "Data escolhida")], []
        for nome in sel_cmp:
            alvo = pd.Timestamp(base_j) - pd.Timedelta(days=prazos_cmp[nome])
            cand = [d for d in datas_j if pd.Timestamp(d) <= alvo]
            if cand:
                datas_plot.append((pd.Timestamp(cand[-1]), nome))
            else:
                faltam.append(nome)
        datas_plot += [(pd.Timestamp(d), "") for d in extra]
        fig = go.Figure()
        for d, rot in datas_plot:
            fig.add_trace(go.Scatter(x=anos, y=curva_juros(pj, d, tc, anos), mode="lines",
                                     name=f"{d.strftime('%d/%m/%Y')} {rot}".strip(), line=dict(width=4 if rot == "Data escolhida" else 2)))
        fig.update_layout(title=f"Curva {tc_rot.lower()} (% a.a.)", height=460, yaxis_tickformat=".2%", xaxis_title="Prazo (anos)",
                          legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=20, l=10, r=10))
        st.plotly_chart(fig, width="stretch")
        if faltam:
            st.caption("Sem histórico suficiente para: " + ", ".join(faltam) + ".")
        vert = [v for v in (0.5, 1, 2, 3, 5, 10, 15, 20, 30) if v <= max_anos]
        tab_j = pd.DataFrame({f"{d.strftime('%d/%m/%Y')} {rot}".strip(): curva_juros(pj, d, tc, vert) * 100 for d, rot in datas_plot},
                             index=[f"{v:g} ano(s)" for v in vert])
        base_col = tab_j.columns[0]
        for c_ in list(tab_j.columns[1:]):
            tab_j[f"Δ vs {c_} (bps)"] = (tab_j[base_col] - tab_j[c_]) * 100
        st.dataframe(tab_j.round(2), width="stretch")
        st.subheader("Histórico de vértices")
        v_hist = st.multiselect("Vértices (anos)", [0.5, 1, 2, 3, 5, 10, 15, 20, 30], default=[1, 2, 5, 10], key="jur_vert")
        if v_hist:
            hist = pd.DataFrame({"data": [pd.Timestamp(d) for d in datas_j]})
            for v in v_hist:
                hist[f"{v:g}a"] = [curva_juros(pj, d, tc, [v])[0] for d in datas_j]
            fig = go.Figure([go.Scatter(x=hist["data"], y=hist[f"{v:g}a"], mode="lines+markers", name=f"{v:g} ano(s)") for v in v_hist])
            fig.update_layout(title=f"{tc_rot}: vértices ao longo do tempo (% a.a.)", height=400, yaxis_tickformat=".2%",
                              legend=dict(orientation="h", y=-0.2), margin=dict(t=40, b=20, l=10, r=10))
            st.plotly_chart(fig, width="stretch")
            st.download_button("⬇️ Histórico (CSV)", hist.assign(data=hist["data"].dt.date).to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"),
                               file_name=f"ettj_{tc}_historico.csv", mime="text/csv", key="jur_csv")
        st.caption("Curvas reconstruídas pelos parâmetros Svensson divulgados pela ANBIMA. Inflação implícita = (1 + prefixada) ÷ (1 + real) − 1.")

# ================================================================== macro
with aba_mac:
    macro_aba.render(pasta_macro(), excel_bytes)

# ================================================================== metodologia
with aba_met:
    st.markdown("""
**Fontes.** Tráfego: ANTT (federais, veículos equivalentes por praça e mês) e ARTESP (SP, tráfego diário agregado por mês).
Financeiro: CVM Dados Abertos (DFP/ITR) de cada SPE com registro, lido com o mesmo código do Price Discovery Scanning
(anual, trimestral por diferença de acumulados e LTM). Debêntures: SND (cadastro) e ANBIMA (taxa indicativa, duration).
Acionistas: Formulário de Referência (posição acionária) e curadoria. Juros: ETTJ ANBIMA. Macro: IBGE e BCB.

**Ajustes de concessão (cálculo próprio).**
- *Receita líquida ajustada*: receita líquida sem a receita de construção (ICPC 01). Sim — a receita líquida exibida já é
  ajustada pela receita de construção; o custo de construção sai do custo e a margem de construção sai do EBITDA.
- *EBITDA ajustado*: EBIT + D&A − margem de construção + constituição da provisão de manutenção (ajuste não caixa da
  DFC). Linhas mistas "provisão e atualização monetária" entram inteiras (a DFC não separa); atualização/AVP pura fica de fora.
- *EBITDA ajustado pós-arrendamento*: EBITDA ajustado − pagamentos de arrendamento (principal + juros, DFC).
- *Capex de manutenção*: pagamento/utilização da provisão de manutenção na DFC. Sem a linha, aproximado por saldo
  inicial + ajuste da DFC (0 se não identificado) − saldo final, marcado como aproximado.
- *ICSD*: (EBITDA ajustado − arrendamento − IR/CS pagos − capex de manutenção) ÷ (juros + principal pagos). Segunda
  versão sem o pagamento da provisão.
- *DL/EBITDA*: com arrendamento ÷ EBITDA ajustado (IFRS 16) e sem arrendamento ÷ EBITDA ajustado pós-arrendamento.
- *Custo da dívida*: debêntures ponderadas pelo saldo (emissão = escritura; mercado = taxa indicativa ANBIMA), spread
  sobre o soberano no mesmo prazo; custo implícito = juros pagos ÷ dívida bruta média (toda a dívida, regime de caixa).

**Limites.** SPEs sem registro na CVM não têm financeiro. Consolidados por grupo somam SPEs (sem holding, sem
eliminações). PPPs (concessão patrocinada) têm contabilidade diferente e entram com aviso. Grupo e acionistas marcados
"validar" ainda não têm documento.
""")
