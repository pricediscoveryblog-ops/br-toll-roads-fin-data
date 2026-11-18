"""BR Toll Roads Fin Data — Price Discovery.

Lê os Parquet publicados pelo coletor (repo público pricediscoveryblog-ops/coletor-concessoes, branch dados).
Fontes: ANTT (federais) e ARTESP (SP). Cada número indica a fonte; métricas calculadas aqui são marcadas.
"""
import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

st.set_page_config(page_title="BR Toll Roads Fin Data", layout="wide")
BASE = os.environ.get("DADOS_BASE", "https://raw.githubusercontent.com/pricediscoveryblog-ops/coletor-concessoes/dados/")
REGIOES = {"N": "Norte", "NE": "Nordeste", "CO": "Centro-Oeste", "SE": "Sudeste", "S": "Sul"}


@st.cache_data(ttl=6 * 3600, show_spinner="Carregando dados…")
def ler(nome: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(BASE + nome + ".parquet")
    except Exception:  # noqa: BLE001
        return pd.DataFrame()


@st.cache_data(ttl=6 * 3600)
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


def anual_antt(nome: str) -> pd.DataFrame:
    d = ler(nome)
    if d.empty:
        return d
    return d[d.id_concessao != ""].groupby(["id_concessao", "ano"], as_index=False)["valor"].sum()


def metricas(cad: pd.DataFrame, traf: pd.DataFrame) -> pd.DataFrame:
    """Métricas por SPE (cálculo próprio sobre dados ANTT/ARTESP)."""
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
        # crescimento em base comparável: só praças com 12 meses de dado nas duas janelas
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
                       "cresc_ltm_mesmas_pracas": cresc_mp, "pracas_novas_ltm": len(set(j1.praca) - set(j0.praca)) if len(j0) else np.nan,
                       "cagr_3a": cagr3,
                       "pct_pesados": ltm.veic_pesado.sum() / ltm.veic_total.sum() if ltm.veic_total.sum() else np.nan,
                       "pior_queda_anual": pior})
    m = cad.merge(pd.DataFrame(linhas), left_on="id", right_on="id_concessao", how="left")
    km = pd.to_numeric(m.km_cadastro, errors="coerce")
    m["vdm_eq_por_km"] = m.veic_eq_ltm / 365 / km
    rec, cus, inv = anual_antt("receita_pedagio"), anual_antt("custos_operacionais"), anual_antt("investimentos")
    if not rec.empty:
        ano = int(rec.ano.max())
        for nome, d in (("receita", rec), ("custos", cus), ("invest", inv)):
            u = d[d.ano == ano].set_index("id_concessao")["valor"] if not d.empty else pd.Series(dtype=float)
            m[f"{nome}_{ano}"] = m["id"].map(u)
        traf_ano = traf[traf.mes.dt.year == ano].groupby("id_concessao").veic_eq.sum()
        m[f"tarifa_media_{ano}"] = m[f"receita_{ano}"] / m["id"].map(traf_ano)
        m[f"custos_sobre_receita_{ano}"] = m[f"custos_{ano}"] / m[f"receita_{ano}"]
        m[f"invest_sobre_receita_{ano}"] = m[f"invest_{ano}"] / m[f"receita_{ano}"]
        m[f"receita_por_km_{ano}"] = m[f"receita_{ano}"] / km
    m["regiao_nome"] = m.regiao.map(REGIOES)
    return m


def baixar_botao(df: pd.DataFrame, nome: str) -> None:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.to_excel(w, index=False)
    c1, c2 = st.columns(2)
    c1.download_button("Baixar CSV", df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig"), f"{nome}.csv")
    c2.download_button("Baixar XLSX", buf.getvalue(), f"{nome}.xlsx")


# ------------------------------------------------------------------ dados
cad = ler("cadastro")
traf = trafego_mensal()
if cad.empty or traf.empty:
    st.error("Não foi possível carregar os dados do coletor. Tente novamente em alguns minutos.")
    st.stop()
cad["grupo"] = cad.grupo.replace("", "(não identificado)")
met = metricas(cad, traf)
nomes = dict(zip(cad.id, cad.nome))

st.title("BR Toll Roads Fin Data")
st.caption("Concessões rodoviárias — federais (ANTT) e SP (ARTESP). Fontes: dados abertos da ANTT e da ARTESP, "
           "coletados semanalmente. Grupo controlador marcado 'validar' vem de conhecimento prévio, ainda sem documento.")

with st.sidebar:
    st.header("Filtros")
    f_ag = st.multiselect("Agência", sorted(cad.agencia.unique()), default=sorted(cad.agencia.unique()))
    f_reg = st.multiselect("Região", list(REGIOES.values()), default=list(REGIOES.values()))
    f_gr = st.multiselect("Grupo controlador", sorted(cad.grupo.unique()))
    so_ativas = st.checkbox("Só SPEs com tráfego nos últimos 12 meses", value=True)
sel = met[met.agencia.isin(f_ag) & met.regiao_nome.isin(f_reg)]
if f_gr:
    sel = sel[sel.grupo.isin(f_gr)]
if so_ativas:
    corte = traf.mes.max() - pd.DateOffset(months=12)
    sel = sel[sel.ultimo_mes > corte]

aba_mapa, aba_op, aba_comp = st.tabs(["Visão geral e mapa", "Operacional por SPE", "Comparar SPEs"])

# ------------------------------------------------------------------ mapa
with aba_mapa:
    c = st.columns(4)
    c[0].metric("SPEs no filtro", len(sel))
    c[1].metric("Veíc. equivalentes LTM (mi)", f"{sel.veic_eq_ltm.sum() / 1e6:,.0f}")
    c[2].metric("Grupos", sel.grupo.nunique())
    c[3].metric("Último mês com dado", traf.mes.max().strftime("%m/%Y"))
    pr = ler("pracas_antt")
    if not pr.empty:
        pr = pr[pr.id_concessao.isin(sel.id) & pr.latitude.notna()].merge(
            cad[["id", "nome", "grupo"]], left_on="id_concessao", right_on="id")
        ltm_praca = traf[traf.mes > traf.mes.max() - pd.DateOffset(months=12)].groupby(["id_concessao", "praca"]).veic_eq.sum()
        pr["veic_eq_ltm"] = [ltm_praca.get((i, p), 0) for i, p in zip(pr.id_concessao, pr.praca_de_pedagio)]
        grupos = sorted(pr.grupo.unique())
        paleta = [[31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40], [148, 103, 189], [140, 86, 75],
                  [227, 119, 194], [127, 127, 127], [188, 189, 34], [23, 190, 207]]
        cor = {g: paleta[i % len(paleta)] for i, g in enumerate(grupos)}
        pr["cor"] = pr.grupo.map(cor)
        pr["raio"] = 4000 + np.sqrt(pr.veic_eq_ltm) * 4
        st.pydeck_chart(pdk.Deck(
            map_style=None,
            initial_view_state=pdk.ViewState(latitude=-17, longitude=-50, zoom=3.6),
            layers=[pdk.Layer("ScatterplotLayer", pr, get_position="[longitude, latitude]", get_fill_color="cor",
                              get_radius="raio", pickable=True, opacity=0.8)],
            tooltip={"text": "{nome} ({grupo})\n{praca_de_pedagio} — {rodovia}/{uf}\nVeíc. eq. LTM: {veic_eq_ltm}"}))
        st.caption("Praças federais com coordenadas da ANTT; tamanho = veículos equivalentes LTM. "
                   "Cores por grupo: " + " · ".join(grupos) + ". Praças de SP entram no mapa quando incorporarmos os KMZ da ARTESP.")
    cols = ["nome", "agencia", "uf_principal", "regiao_nome", "grupo", "grupo_fonte", "km_cadastro", "modalidade", "lote_obs", "obs"]
    st.dataframe(sel[cols], hide_index=True, width="stretch")

# ------------------------------------------------------------------ operacional
with aba_op:
    op = st.selectbox("SPE", sel.id.tolist(), format_func=lambda i: nomes.get(i, i))
    t = traf[traf.id_concessao == op]
    if t.empty:
        st.info("Sem tráfego publicado para esta SPE.")
    else:
        fonte = t.fonte.iloc[0]
        m = t.groupby("mes", as_index=False)[["veic_eq", "veic_total", "veic_pesado"]].sum().sort_values("mes")
        m["ltm"] = m.veic_eq.rolling(12).sum()
        m["var_aa"] = m.veic_eq.pct_change(12)
        st.subheader(f"{nomes[op]} — tráfego mensal (fonte: {fonte})")
        st.plotly_chart(px.line(m, x="mes", y=["veic_eq", "veic_total"], labels={"value": "veículos", "mes": ""}),
                        width="stretch")
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.line(m, x="mes", y="ltm", title="Veículos equivalentes — 12 meses acumulados"),
                        width="stretch")
        c2.plotly_chart(px.bar(m.dropna(subset=["var_aa"]), x="mes", y="var_aa", title="Variação a/a (mês vs mesmo mês do ano anterior)"),
                        width="stretch")
        ult = m.mes.max()
        pp = t[t.mes > ult - pd.DateOffset(months=12)].groupby("praca", as_index=False)[["veic_eq", "veic_total", "veic_pesado"]].sum()
        pp["pct_pesados"] = pp.veic_pesado / pp.veic_total
        st.markdown("**Praças — últimos 12 meses**")
        st.dataframe(pp.sort_values("veic_eq", ascending=False), hide_index=True, width="stretch")
        if fonte == "ANTT":
            rec, cus, inv = anual_antt("receita_pedagio"), anual_antt("custos_operacionais"), anual_antt("investimentos")
            fin = rec[rec.id_concessao == op].rename(columns={"valor": "receita_pedagio"})
            for nome, d in (("custos_operacionais", cus), ("investimentos", inv)):
                fin = fin.merge(d[d.id_concessao == op].rename(columns={"valor": nome}), on=["id_concessao", "ano"], how="outer")
            if not fin.empty:
                eq_ano = t.groupby(t.mes.dt.year).veic_eq.sum()
                fin["tarifa_media_por_veic_eq"] = fin.receita_pedagio / fin.ano.map(eq_ano)
                st.markdown("**Dados econômicos reportados à ANTT (R$, anual)** — tarifa média = receita ÷ veículos equivalentes (cálculo próprio)")
                st.plotly_chart(px.bar(fin.sort_values("ano"), x="ano", y=["receita_pedagio", "custos_operacionais", "investimentos"],
                                       barmode="group", labels={"value": "R$", "ano": ""}), width="stretch")
                st.dataframe(fin.sort_values("ano"), hide_index=True, width="stretch")
        baixar_botao(m, f"trafego_{op}")

# ------------------------------------------------------------------ comparar
with aba_comp:
    st.markdown("Métricas calculadas aqui a partir dos dados da ANTT/ARTESP. LTM = últimos 12 meses com dado da SPE. "
                "Percentil dentro do grupo de pares escolhido. **cresc_ltm_mesmas_pracas** compara só praças com 12 meses "
                "de dado nas duas janelas — isola a entrada de novas praças/pórticos (ex.: RIOSP a partir de dez/2025).")
    chave = st.selectbox("Grupo de pares (chave de comparação)", ["regiao_nome", "uf_principal", "agencia", "grupo", "(todas)"],
                         format_func=lambda c: {"regiao_nome": "Região", "uf_principal": "UF", "agencia": "Agência",
                                                "grupo": "Grupo controlador", "(todas)": "Todas"}[c])
    num = [c for c in sel.columns if c in ("veic_eq_ltm", "cresc_ltm", "cresc_ltm_mesmas_pracas", "pracas_novas_ltm", "cagr_3a", "pct_pesados", "vdm_eq_por_km",
                                           "pior_queda_anual") or c.startswith(("receita_", "tarifa_media", "custos_sobre", "invest_sobre"))]
    comp = sel[["id", "nome", "grupo", "regiao_nome", "uf_principal", "agencia", "ultimo_mes"] + num].copy()
    if chave != "(todas)":
        for c in num:
            comp[f"pctl_{c}"] = comp.groupby(chave)[c].rank(pct=True)
    st.dataframe(comp.drop(columns="id"), hide_index=True, width="stretch")
    baixar_botao(comp.drop(columns="id"), "comparacao_spes")

    c1, c2, c3 = st.columns(3)
    x = c1.selectbox("Eixo X", num, index=num.index("cresc_ltm_mesmas_pracas") if "cresc_ltm_mesmas_pracas" in num else 0)
    y = c2.selectbox("Eixo Y", num, index=num.index("vdm_eq_por_km") if "vdm_eq_por_km" in num else 0)
    cor = c3.selectbox("Cor", ["regiao_nome", "grupo", "agencia", "uf_principal"])
    st.plotly_chart(px.scatter(comp, x=x, y=y, color=cor, hover_name="nome", size=comp.veic_eq_ltm.fillna(0).clip(lower=1)),
                    width="stretch")

    st.markdown("**Tráfego base 100**")
    escolha = st.multiselect("SPEs", comp.id.tolist(), format_func=lambda i: nomes.get(i, i),
                             default=comp.sort_values("veic_eq_ltm", ascending=False).id.head(5).tolist())
    if escolha:
        b = traf[traf.id_concessao.isin(escolha)].groupby(["id_concessao", "mes"], as_index=False).veic_eq.sum()
        b["ltm"] = b.groupby("id_concessao").veic_eq.transform(lambda s: s.rolling(12).sum())
        b = b.dropna(subset=["ltm"])
        inicio = st.slider("Início da base", min_value=b.mes.min().to_pydatetime(), max_value=b.mes.max().to_pydatetime(),
                           value=max(b.mes.min(), b.mes.max() - pd.DateOffset(years=5)).to_pydatetime(), format="MM/YYYY")
        b = b[b.mes >= inicio]
        b["base100"] = b.groupby("id_concessao").ltm.transform(lambda s: s / s.iloc[0] * 100)
        b["SPE"] = b.id_concessao.map(nomes)
        st.plotly_chart(px.line(b, x="mes", y="base100", color="SPE", labels={"mes": "", "base100": "LTM, base 100"}),
                        width="stretch")
        st.caption("Série LTM (12 meses acumulados) para neutralizar sazonalidade. SPEs que entram depois da data-base começam no 1º LTM disponível.")
