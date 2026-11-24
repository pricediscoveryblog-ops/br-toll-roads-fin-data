"""Gera geo/tracados.geojson: a geometria (traçado) de cada concessão do cadastro, para o mapa do app.

Fontes (todas públicas):
- Federais (ANTT): trechos concedidos da ANTT (dataset "trecho-concedido", já baixado pelo coletor em
  coletor-concessoes/dados/trechos_antt.parquet: rodovia, UF, km inicial e final de cada SPE) cortados sobre a
  geometria do SNV do DNIT (Sistema Nacional de Viação, segmentos com km inicial/final). Por padrão lê o GeoJSON do
  SNV 202604A espelhado em github.com/lucaasmeiraf/snv-geojson (cópia do arquivo do DNIT); `SNV_URL` troca a fonte.
- Contratos federais encerrados e BRs delegadas a MG: geo/complementos_tracados.csv (critério documentado por linha).
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
            por_chave[(str(p["vl_br"]).zfill(3), p["sg_uf"])].append(
                {"ki": float(p["vl_km_inic"]), "kf": float(p["vl_km_fina"]), "tipo": p.get("sg_tipo_tr", "B"),
                 "adm": p.get("ds_tipo_ad", ""), "cod": p.get("vl_codigo", ""), "g": g})
        # orienta cada segmento do eixo principal pelo vizinho: o fim do segmento encosta no início do próximo
        self.seg = {}
        for k, lst in por_chave.items():
            eixo = sorted([s for s in lst if s["tipo"] == "B"], key=lambda s: s["ki"])
            for i, s in enumerate(eixo):
                a, b = s["g"].coords[0], s["g"].coords[-1]
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

    snv = SNV(json.loads(baixar(args.snv, timeout=600)))
    cad = pd.read_csv(io.BytesIO(baixar(URL_CADASTRO)), sep=";", dtype=str, keep_default_na=False)
    tr = pd.read_parquet(io.BytesIO(baixar(URL_TRECHOS)))
    fonte_snv = f"SNV {snv.versao} (DNIT)"

    feats = []
    # 1) federais ANTT: só pista principal das BRs (acessos/alças ficam de fora; rodovias PR-xxx não estão no SNV)
    tr = tr[tr.tipo_de_rodovia.eq("Federal")].copy()
    tr["uf_sigla"] = tr.uf.astype(str).map(UF_ANTT).fillna(tr.uf.astype(str))
    tr["a"] = tr[["km_m_inicial", "km_m_final"]].min(axis=1)
    tr["b"] = tr[["km_m_inicial", "km_m_final"]].max(axis=1)
    tr = tr.drop_duplicates(["id_concessao", "rodovia", "uf_sigla", "a", "b"])
    for r in tr.itertuples():
        br = br_num(r.rodovia)
        if not br:
            continue
        linhas = [ln.simplify(TOL) for ln in snv.trecho(br, r.uf_sigla, r.a, r.b)]
        if not linhas:
            log.append(f"sem geometria SNV: {r.id_concessao} BR-{br}/{r.uf_sigla} km {r.a}-{r.b}")
            continue
        feats.append(feature(r.id_concessao, f"BR-{br}", r.uf_sigla, r.a, r.b, linhas,
                             f"ANTT trechos concedidos (km) × {fonte_snv}"))
    # 2) complementos documentados (contratos encerrados, BRs delegadas)
    comp = pd.read_csv(COMPLEMENTOS, sep=";", dtype={"km_ini": float, "km_fim": float})
    for r in comp.itertuples():
        br = br_num(r.rodovia)
        excl = "Concessão Federal" if r.id_concessao.startswith("M-") else None
        linhas = [ln.simplify(TOL) for ln in snv.trecho(br, r.uf, r.km_ini, r.km_fim, excluir_adm=excl)]
        if linhas:
            feats.append(feature(r.id_concessao, f"BR-{br}", r.uf, r.km_ini, min(r.km_fim, max(
                s["kf"] for s in snv.seg[(br, r.uf)])), linhas, f"{fonte_snv}; critério: {r.criterio}"))
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
