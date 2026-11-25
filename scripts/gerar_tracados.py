"""Gera geo/tracados.geojson: a geometria (traçado) de cada concessão do cadastro, para o mapa do app.

Fontes (todas públicas):
- Federais (ANTT): trechos concedidos da ANTT (dataset "trecho-concedido", já baixado pelo coletor em
  coletor-concessoes/dados/trechos_antt.parquet: rodovia, UF, km inicial e final de cada SPE) cortados sobre a
  geometria do SNV do DNIT (Sistema Nacional de Viação, segmentos com km inicial/final). Por padrão lê o GeoJSON do
  SNV 202604A espelhado em github.com/lucaasmeiraf/snv-geojson (cópia do arquivo do DNIT); `SNV_URL` troca a fonte.
- Rodovias estaduais dos lotes do PR (PR-xxx delegadas à União, linhas "Estadual" da ANTT): km da ANTT cortados sobre o
  SRE do DER-PR (camada der:sistema_rodoviario_lin_p31982 do GeoServer do Paraná, a mesma do GeoDER, com código do
  trecho e km inicial/final). Baixada a cada execução; se o GeoServer falhar, usa a cópia geo/der_pr_trechos.geojson.
- Contornos, acessos e travessias federais dos lotes do PR (linhas "Outros" da ANTT): segmentos auxiliares do SNV
  (código do trecho, ex. 376CPR1 = Contorno Norte de Maringá), pelos critérios de geo/complementos_tracados.csv.
- Contratos federais encerrados, BRs delegadas a MG e lotes estaduais de MG (Artemig): geo/complementos_tracados.csv
  (critério e fonte por linha). Coluna `base`: `snv` corta o SNV/DNIT; `dermg` corta a malha estadual do DER-MG
  (geo/der_mg_trechos.geojson, cópia da camada IDE:ide_0401_mg_rodovias_lin da IDE-Sisema, mesmo esquema do SNV com
  código SRE e km inicial/final por segmento); `derpr` corta o SRE do DER-PR; `snvaux` corta os segmentos auxiliares do
  SNV (contornos/acessos/travessias/variantes) cujo código começa pelo prefixo da coluna `rodovia`; `ignorar` descarta
  um trecho federal da ANTT cujo km não corresponde ao eixo principal da BR no SNV (substituído por outra linha).
- SP (ARTESP): KMZ oficiais por lote do dataset "malha-rodoviaria" do portal de dados abertos da ARTESP.
  O portal só é acessível de redes abertas (GitHub Actions); se falhar, os lotes de SP ficam de fora e o app usa o
  ponto aproximado do cadastro.

Uso:  python scripts/gerar_tracados.py [--sem-artesp] [--snv caminho_ou_url]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import sys
import unicodedata
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

import pandas as pd
from shapely.geometry import LineString, MultiLineString, shape
from shapely.ops import linemerge, substring

RAIZ = Path(__file__).resolve().parents[1]
SAIDA = RAIZ / "geo" / "tracados.geojson"
COMPLEMENTOS = RAIZ / "geo" / "complementos_tracados.csv"
DER_MG = RAIZ / "geo" / "der_mg_trechos.geojson"   # cópia da camada do DER-MG (IDE-Sisema) com as estaduais dos lotes
DER_PR = RAIZ / "geo" / "der_pr_trechos.geojson"   # cópia (só as rodovias usadas) do SRE do DER-PR, reserva se o WFS cair
URL_DER_PR = os.environ.get(
    "DER_PR_URL", "https://geoserver.pr.gov.br/geoserver/ows?service=wfs&version=2.0.0&request=GetFeature"
    "&typeNames=der:sistema_rodoviario_lin_p31982&outputFormat=application/json&srsName=EPSG:4674")
AUX_TIPO = {"A": "acesso", "C": "contorno", "U": "travessia urbana", "V": "variante"}
RAW = "https://raw.githubusercontent.com/pricediscoveryblog-ops/coletor-concessoes/"
URL_TRECHOS = os.environ.get("TRECHOS_URL", RAW + "dados/trechos_antt.parquet")
URL_CADASTRO = os.environ.get("CADASTRO_URL", RAW + "main/cadastro/concessoes.csv")
URL_SNV = os.environ.get(
    "SNV_URL", "https://raw.githubusercontent.com/lucaasmeiraf/snv-geojson/master/GeoJSON%20-%20SNV_202604A.json")
API_ARTESP = "https://dadosabertos.artesp.sp.gov.br/api/3/action/package_show?id=malha-rodoviaria"
TOL = 0.0008       # simplificação (graus, ~90 m): mantém o desenho da via e deixa o arquivo leve
CASAS = 5

# Código de UF usado pela ANTT na base de trechos (conferido contra a geometria do SNV: cada código cai no estado)
UF_ANTT = {"55": "RO", "59": "PA", "61": "TO", "70": "BA", "71": "MG", "72": "ES", "73": "RJ", "74": "SP", "75": "PR",
           "76": "SC", "77": "RS", "78": "MS", "79": "MT", "80": "GO", "81": "DF"}


def baixar(url: str, timeout: int = 180) -> bytes:
    if not url.startswith("http"):
        return Path(url).read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (price-discovery tracados)"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().upper()
    return re.sub(r"[^A-Z0-9]+", " ", s).strip()


# ------------------------------------------------------------------ SNV (DNIT)
class SNV:
    """Segmentos do SNV por (BR, UF), com a geometria orientada no sentido do km crescente."""

    def __init__(self, geojson: dict):
        self.versao = ""
        por_chave = defaultdict(list)
        for f in geojson["features"]:
            p = f["properties"]
            if not f.get("geometry"):
                continue
            g = shape(f["geometry"])
            if g.is_empty:
                continue
            if g.geom_type == "MultiLineString":
                m = linemerge(g)
                g = m if m.geom_type == "LineString" else max(m.geoms, key=lambda x: x.length)
            self.versao = p.get("versao_snv", self.versao)
            ini = (p["xi"], p["yi"]) if p.get("xi") is not None and p.get("yi") is not None else None
            por_chave[(str(p["vl_br"]).zfill(3), p["sg_uf"])].append(
                {"ki": float(p["vl_km_inic"]), "kf": float(p["vl_km_fina"]), "tipo": p.get("sg_tipo_tr", "B"),
                 "adm": p.get("ds_tipo_ad", ""), "cod": p.get("vl_codigo", ""), "g": g, "ini": ini})
        # orienta cada segmento do eixo principal pelo vizinho: o fim do segmento encosta no início do próximo
        self.seg = {}
        for k, lst in por_chave.items():
            eixo = sorted([s for s in lst if s["tipo"] == "B"], key=lambda s: s["ki"])
            for i, s in enumerate(eixo):
                a, b = s["g"].coords[0], s["g"].coords[-1]
                if s["ini"] is not None:  # ponto inicial do trecho na base (DER-PR): o início do km fica nele
                    if math.dist(b, s["ini"]) < math.dist(a, s["ini"]):
                        s["g"] = LineString(list(s["g"].coords)[::-1])
                    continue
                viz = eixo[i + 1]["g"] if i + 1 < len(eixo) else (eixo[i - 1]["g"] if i > 0 else None)
                if viz is not None:
                    pv = [viz.coords[0], viz.coords[-1]]
                    da = min(math.dist(a, q) for q in pv)
                    db = min(math.dist(b, q) for q in pv)
                    proximo = i + 1 < len(eixo)
                    # se o vizinho é o PRÓXIMO, o fim (b) deve estar mais perto dele; se é o anterior, o início (a)
                    inverter = (da < db) if proximo else (db < da)
                    if inverter:
                        s["g"] = LineString(list(s["g"].coords)[::-1])
            self.seg[k] = eixo

    def trecho(self, br: str, uf: str, ki: float, kf: float, excluir_adm: str | None = None) -> list[LineString]:
        ki, kf = min(ki, kf), max(ki, kf)
        out = []
        for s in self.seg.get((br.zfill(3), uf), []):
            if excluir_adm and s["adm"] == excluir_adm:
                continue
            a, b = max(ki, s["ki"]), min(kf, s["kf"])
            if b - a <= 0.05:
                continue
            ext = s["kf"] - s["ki"]
            if ext <= 0:
                continue
            g = s["g"]
            if a <= s["ki"] + 1e-6 and b >= s["kf"] - 1e-6:
                out.append(g)
            else:  # corte proporcional ao km dentro do segmento
                p = substring(g, (a - s["ki"]) / ext, (b - s["ki"]) / ext, normalized=True)
                if p.geom_type == "LineString" and not p.is_empty:
                    out.append(p)
        return out


def gms(txt) -> float | None:
    """'-24°33′26,26″' -> graus decimais (latitude/longitude do início do trecho no SRE do DER-PR)."""
    m = re.match(r"\s*(-?)\s*(\d+)\s*°\s*(\d+)\s*[′']\s*([\d.,]+)", str(txt or ""))
    if not m:
        return None
    v = int(m.group(2)) + int(m.group(3)) / 60 + float(m.group(4).replace(",", ".")) / 3600
    return -v if m.group(1) else v


def pista_esquerda(trecho: str) -> bool:
    """Trechos duplicados vêm em duas pistas (D/E): desenha só uma (a direita), como no SNV."""
    return trecho[3:4] == "E" or bool(re.match(r"^\d{3}[A-Z]PR\d+E", trecho))


def der_pr(rods: set[str], log: list) -> SNV | None:
    """SRE do DER-PR como base no mesmo esquema do SNV: chave (rodovia, 'PR'), km inicial/final por trecho. Fica só o
    eixo das rodovias estaduais (inclui trechos coincidentes com BR, municipais e não pavimentados no meio do eixo);
    saem a pista esquerda dos duplicados, os acessos/contornos/travessias/variantes das BRs (códigos ###[ACUV]PR) e os
    trechos coincidentes com BR (###BPR), que usam a quilometragem da BR e se sobrepõem aos km da rodovia estadual."""
    gd = None
    try:
        bruto = json.loads(baixar(URL_DER_PR, timeout=300))
        feats = []
        for f in bruto["features"]:
            p, g = f["properties"], f.get("geometry")
            t = str(p.get("trecho") or "")
            if not g or not t or str(p.get("rod_num", "")).zfill(3) not in rods:
                continue
            geo = shape(g).simplify(0.0001)
            if geo.is_empty:
                continue
            feats.append({"type": "Feature", "properties": {
                "trecho": t, "rod_num": int(p["rod_num"]), "km_inicial": p["km_inicial"], "km_final": p["km_final"],
                "tipojuris": p.get("tipojuris"), "concessao": p.get("concessao"),
                "xi": gms(p.get("longitudei")), "yi": gms(p.get("latitudei"))},
                "geometry": json.loads(json.dumps(geo.__geo_interface__))})
        if not feats:
            raise ValueError("WFS sem trechos das rodovias pedidas")
        for f in feats:
            gm = f["geometry"]
            linhas = gm["coordinates"] if gm["type"] == "MultiLineString" else [gm["coordinates"]]
            gm.update(type="MultiLineString", coordinates=[[[round(x, CASAS), round(y, CASAS)] for x, y in ln]
                                                           for ln in linhas])
        feats.sort(key=lambda f: (f["properties"]["rod_num"], f["properties"]["km_inicial"], f["properties"]["trecho"]))
        gd = {"type": "FeatureCollection", "fonte": "DER-PR, GeoServer do Paraná (der:sistema_rodoviario_lin_p31982)",
              "features": feats}
        DER_PR.write_text(json.dumps(gd, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        log.append(f"DER-PR: {len(feats)} trechos do SRE (WFS) para {len(rods)} rodovias; cópia em {DER_PR.name}")
    except Exception as e:  # noqa: BLE001
        if DER_PR.exists():
            gd = json.loads(DER_PR.read_text(encoding="utf-8"))
            log.append(f"DER-PR: WFS indisponível ({e}); usando a cópia {DER_PR.name}")
        else:
            log.append(f"DER-PR: WFS indisponível ({e}) e sem cópia; estaduais do PR sem traçado")
            return None
    sel = [f for f in gd["features"] if not pista_esquerda(f["properties"]["trecho"])
           and not re.match(r"^\d{3}[ABCUV]PR", f["properties"]["trecho"])]
    return SNV({"features": [{"geometry": f["geometry"], "properties": {
        "vl_br": str(f["properties"]["rod_num"]).zfill(3), "sg_uf": "PR", "vl_km_inic": f["properties"]["km_inicial"],
        "vl_km_fina": f["properties"]["km_final"], "sg_tipo_tr": "B", "ds_tipo_ad": f["properties"].get("tipojuris", ""),
        "vl_codigo": f["properties"]["trecho"], "versao_snv": "", "xi": f["properties"].get("xi"),
        "yi": f["properties"].get("yi")}} for f in sel]})


def br_num(rod: str) -> str | None:
    m = re.match(r"^\s*BR[-\s]*(\d{1,3})", str(rod), re.I)
    return m.group(1).zfill(3) if m else None


def km_de(linhas: list[LineString]) -> float:
    """Extensão aproximada em km (graus → km na latitude média)."""
    tot = 0.0
    for ln in linhas:
        c = list(ln.coords)
        for (x1, y1), (x2, y2) in zip(c[:-1], c[1:]):
            lat = math.radians((y1 + y2) / 2)
            tot += 111.32 * math.hypot((x2 - x1) * math.cos(lat), y2 - y1)
    return tot


# ------------------------------------------------------------------ ARTESP (KMZ por lote)
RE_LINHA = re.compile(rb"<(?:[\w.-]+:)?LineString\b.*?</(?:[\w.-]+:)?LineString>", re.S)
RE_COORD = re.compile(rb"<(?:[\w.-]+:)?coordinates[^>]*>(.*?)</(?:[\w.-]+:)?coordinates>", re.S)


def linhas_kml(kml: bytes) -> list[LineString]:
    """Extrai todas as LineString de um KML. Leitura por expressão regular, e não por parser XML, porque alguns
    KMZ da ARTESP trazem prefixos de namespace não declarados (XML inválido para o parser)."""
    out = []
    for bloco in RE_LINHA.findall(kml):
        for txt in RE_COORD.findall(bloco):
            pts = []
            for tok in txt.split():
                v = tok.split(b",")
                if len(v) >= 2:
                    try:
                        pts.append((float(v[0]), float(v[1])))
                    except ValueError:
                        pass
            if len(pts) >= 2:
                out.append(LineString(pts))
    return out


GRADE = 0.003     # ~300 m: as duas pistas, marginais e alças paralelas caem no mesmo nó
MIN_LEN = 0.004   # descarta pedaços soltos < ~400 m (alças e dispositivos isolados)


def consolidar(linhas: list[LineString]) -> list[LineString]:
    """Os KMZ da ARTESP vêm em centenas de milhares de segmentos (pista a pista, alças, dispositivos). Para o mapa:
    encaixa os vértices numa grade de ~300 m, remove arestas repetidas (pistas paralelas viram uma), junta e simplifica."""
    arestas = set()
    for ln in linhas:
        pts = [(round(x / GRADE) * GRADE, round(y / GRADE) * GRADE) for x, y in ln.coords]
        for a, b in zip(pts[:-1], pts[1:]):
            if a != b:
                arestas.add((a, b) if a < b else (b, a))
    if not arestas:
        return []
    m = linemerge(MultiLineString([LineString(e) for e in arestas]))
    geoms = [m] if m.geom_type == "LineString" else list(m.geoms)
    return [g.simplify(GRADE * 0.8) for g in geoms if g.length >= MIN_LEN]


def tracados_artesp(cad: pd.DataFrame, log: list) -> list[dict]:
    try:
        pac = json.loads(baixar(API_ARTESP, timeout=120))
    except Exception as e:  # noqa: BLE001
        log.append(f"ARTESP: API indisponível ({e}); lotes de SP sem traçado")
        return []
    sp = cad[cad.agencia == "ARTESP"]
    chaves = {}  # nome normalizado -> id do lote
    for i, nomes, nome in zip(sp.id, sp.nomes_fonte, sp.nome):
        for n in [nome] + str(nomes).split("|"):
            n = norm(re.sub(r"^L\d+\s*", "", n))
            if len(n) >= 4:
                chaves[n] = i
    feats = []
    for r in pac["result"]["resources"]:
        nome_r = str(r.get("name", ""))
        if "KMZ" not in nome_r.upper() and not str(r.get("url", "")).lower().endswith((".kmz", ".kml")):
            continue
        alvo = norm(re.sub(r"(?i)\bkmz\b", "", nome_r))
        lote = chaves.get(alvo) or next((v for k, v in sorted(chaves.items(), key=lambda kv: -len(kv[0]))
                                         if k in alvo or alvo in k), None)
        if not lote:
            log.append(f"ARTESP: recurso sem lote no cadastro: {nome_r}")
            continue
        try:
            b = baixar(r["url"], timeout=300)
            kml = b
            if b[:2] == b"PK":
                z = zipfile.ZipFile(io.BytesIO(b))
                kml = z.read(next(n for n in z.namelist() if n.lower().endswith(".kml")))
            linhas = linhas_kml(kml)
        except Exception as e:  # noqa: BLE001
            log.append(f"ARTESP: falha em {nome_r} ({e})")
            continue
        n_bruto = len(linhas)
        linhas = consolidar(linhas)
        if linhas:
            f = feature(lote, "Malha do lote (KMZ ARTESP)", "SP", None, None, linhas,
                        f"ARTESP - dados abertos, malha-rodoviaria: {nome_r}")
            f["properties"]["km_tracado"] = None   # inclui pistas/alças: não é a extensão do lote
            feats.append(f)
            log.append(f"ARTESP: {lote} <- {nome_r}: {n_bruto} segmentos -> {len(linhas)} linhas")
    return feats


# ------------------------------------------------------------------ montagem
def feature(id_: str, rod: str, uf: str, ki, kf, linhas: list[LineString], fonte: str) -> dict:
    # junta segmentos contíguos (os do SNV têm poucos km cada) numa linha contínua: arquivo menor e desenho limpo
    m = linemerge(MultiLineString([ln for ln in linhas if ln.length > 0])) if len(linhas) > 1 else None
    if m is not None and not m.is_empty:
        linhas = [m] if m.geom_type == "LineString" else list(m.geoms)
    coords = [[[round(x, CASAS), round(y, CASAS)] for x, y in ln.coords] for ln in linhas]
    return {"type": "Feature",
            "properties": {"id_concessao": id_, "rodovia": rod, "uf": uf,
                           "km_ini": None if ki is None else round(float(ki), 1),
                           "km_fim": None if kf is None else round(float(kf), 1),
                           "km_tracado": round(km_de(linhas), 1), "fonte": fonte},
            "geometry": {"type": "MultiLineString", "coordinates": coords}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem-artesp", action="store_true")
    ap.add_argument("--snv", default=URL_SNV)
    args = ap.parse_args()
    log: list[str] = []

    snv_json = json.loads(baixar(args.snv, timeout=600))
    snv = SNV(snv_json)
    # segmentos auxiliares do SNV (contornos, acessos, travessias, variantes) por prefixo do código (ex. 376CPR1)
    aux = [f for f in snv_json["features"] if f.get("geometry") and f["properties"].get("sg_tipo_tr", "B") != "B"]
    snv_aux = SNV({"features": [{"geometry": f["geometry"], "properties": {
        **f["properties"], "vl_br": str(f["properties"].get("vl_codigo", ""))[:7], "sg_tipo_tr": "B"}} for f in aux]})
    snv_aux.versao = snv.versao
    cad = pd.read_csv(io.BytesIO(baixar(URL_CADASTRO)), sep=";", dtype=str, keep_default_na=False)
    tr = pd.read_parquet(io.BytesIO(baixar(URL_TRECHOS)))
    fonte_snv = f"SNV {snv.versao} (DNIT)"

    comp = pd.read_csv(COMPLEMENTOS, sep=";", dtype={"km_ini": float, "km_fim": float}, keep_default_na=False)
    ignorar = {(r.id_concessao, br_num(r.rodovia), r.uf, round(r.km_ini, 1), round(r.km_fim, 1))
               for r in comp.itertuples() if r.base == "ignorar"}

    feats = []
    tr["uf_sigla"] = tr.uf.astype(str).map(UF_ANTT).fillna(tr.uf.astype(str))
    tr["a"] = tr[["km_m_inicial", "km_m_final"]].min(axis=1)
    tr["b"] = tr[["km_m_inicial", "km_m_final"]].max(axis=1)
    tr = tr.drop_duplicates(["id_concessao", "tipo_de_rodovia", "rodovia", "uf_sigla", "a", "b"])
    est = tr[tr.tipo_de_rodovia.eq("Estadual") & tr.uf_sigla.eq("PR")].copy()
    # 1) federais ANTT: só pista principal das BRs (acessos/contornos dos lotes do PR entram pelos complementos)
    tr = tr[tr.tipo_de_rodovia.eq("Federal")].copy()
    for r in tr.itertuples():
        br = br_num(r.rodovia)
        if not br:
            continue
        if (r.id_concessao, br, r.uf_sigla, round(r.a, 1), round(r.b, 1)) in ignorar:
            log.append(f"ignorado (complementos): {r.id_concessao} BR-{br}/{r.uf_sigla} km {r.a}-{r.b}")
            continue
        linhas = [ln.simplify(TOL) for ln in snv.trecho(br, r.uf_sigla, r.a, r.b)]
        if not linhas:
            log.append(f"sem geometria SNV: {r.id_concessao} BR-{br}/{r.uf_sigla} km {r.a}-{r.b}")
            continue
        feats.append(feature(r.id_concessao, f"BR-{br}", r.uf_sigla, r.a, r.b, linhas,
                             f"ANTT trechos concedidos (km) × {fonte_snv}"))
    # 1b) estaduais dos lotes do PR (PR-xxx delegadas à União): km da ANTT sobre o SRE do DER-PR
    est["rod3"] = est.rodovia.map(lambda v: (re.search(r"(\d{1,3})\s*$", str(v)) or [None, ""])[1].zfill(3))
    rods_pr = set(est.rod3) | {re.search(r"(\d{1,3})\s*$", r.rodovia).group(1).zfill(3)
                               for r in comp.itertuples() if r.base == "derpr"}
    der_p = der_pr(rods_pr, log) if rods_pr else None
    fonte_derpr = "DER-PR (SRE, GeoServer do Paraná/GeoDER)"
    for r in est.itertuples():
        if der_p is None:
            break
        linhas = [ln.simplify(TOL) for ln in der_p.trecho(r.rod3, "PR", r.a, r.b)]
        if not linhas:
            log.append(f"sem geometria DER-PR: {r.id_concessao} PR-{r.rod3} km {r.a}-{r.b}")
            continue
        feats.append(feature(r.id_concessao, f"PR-{r.rod3}", "PR", r.a, r.b, linhas,
                             f"ANTT trechos concedidos (km, rodovia estadual delegada) × {fonte_derpr}"))
    # 2) complementos documentados (contratos encerrados, BRs delegadas, lotes estaduais de MG, acessos do PR)
    der = None
    if DER_MG.exists():
        gd = json.loads(DER_MG.read_text(encoding="utf-8"))
        der = SNV({"features": [{"geometry": f["geometry"], "properties": {
            "vl_br": f["properties"]["codigo_snv"][:4], "sg_uf": "MG", "vl_km_inic": f["properties"]["km_inicial"],
            "vl_km_fina": f["properties"]["km_final"], "sg_tipo_tr": "B", "ds_tipo_ad": f["properties"]["jurisdicao"],
            "vl_codigo": f["properties"]["codigo_snv"], "versao_snv": ""}} for f in gd["features"]]})
    for r in comp.itertuples():
        excl = r.excluir_adm or None
        if r.base == "ignorar":
            continue
        if r.base == "derpr":
            chave = re.search(r"(\d{1,3})\s*$", r.rodovia).group(1).zfill(3)
            base, rot, fonte = der_p, f"PR-{chave}", fonte_derpr
        elif r.base == "snvaux":
            chave = r.rodovia.strip().upper()
            base, fonte = snv_aux, f"{fonte_snv}, segmentos {chave}*"
            rot = f"BR-{chave[:3]} ({AUX_TIPO.get(chave[3:4], 'acesso')})"
        elif r.base == "dermg":
            num = re.search(r"(\d{1,3})\s*$", r.rodovia).group(1).zfill(3)
            chave, base, rot = num + "E", der, r.rodovia
            fonte = "DER-MG (IDE-Sisema, malha rodoviária estadual)"
        else:
            chave = br_num(r.rodovia)
            base, rot, fonte = snv, f"BR-{chave}", fonte_snv
        if base is None or (chave, r.uf) not in base.seg:
            log.append(f"complemento sem base: {r.id_concessao} {r.rodovia}/{r.uf}")
            continue
        linhas = [ln.simplify(TOL) for ln in base.trecho(chave, r.uf, r.km_ini, r.km_fim, excluir_adm=excl)]
        if linhas:
            kf = min(r.km_fim, max(s["kf"] for s in base.seg[(chave, r.uf)]))
            feats.append(feature(r.id_concessao, rot, r.uf, r.km_ini, kf, linhas, f"{fonte}; critério: {r.criterio}"))
        else:
            log.append(f"complemento sem geometria: {r.id_concessao} {r.rodovia}/{r.uf}")
    # 3) SP
    if not args.sem_artesp:
        feats += tracados_artesp(cad, log)

    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    SAIDA.write_text(json.dumps({"type": "FeatureCollection", "snv": snv.versao, "features": feats},
                                ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    ids = sorted({f["properties"]["id_concessao"] for f in feats})
    print(f"{len(feats)} trechos, {len(ids)} concessões, {SAIDA.stat().st_size / 1e6:.1f} MB -> {SAIDA}")
    print("\n".join(log))
    sem = sorted(set(cad.id) - set(ids))
    print("Sem traçado:", ", ".join(sem) if sem else "nenhuma")


if __name__ == "__main__":
    sys.exit(main())
