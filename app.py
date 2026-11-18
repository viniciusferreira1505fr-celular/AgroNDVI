import io
import json
import os
import re
import shutil
import zipfile
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import rasterio
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.warp import reproject, Resampling
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AgroNDVI", page_icon="🌱", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1450px;}
.hero {padding: 1.25rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#123c2b,#26734d); color: white; margin-bottom: 1rem;}
.hero h1 {margin:0; font-size:2rem; color:white;}
.hero p {margin:.35rem 0 0; opacity:.9;}
.small-note {font-size:.85rem; color: #777;}
</style>
<div class="hero"><h1>🌱 AgroNDVI</h1><p>Monitoramento de vegetação e suporte à pesquisa agrícola</p></div>
""", unsafe_allow_html=True)

st.caption("AgroNDVI 2.0.0 · Plataforma agronômica integrada")

with st.sidebar:
    st.header("Configuração")
    source_mode = st.radio(
        "Origem das imagens",
        ["Sentinel-2 (talhão salvo ou GeoJSON)", "Enviar bandas GeoTIFF"],
        index=0,
    )
    st.divider()
    field_name = st.text_input("Nome da fazenda / talhão / experimento", value="Talhão experimental")
    app_page = st.radio("Área do aplicativo", ["🏠 Painel e relatórios", "🌿 Processar NDVI", "🗺️ Desenhar e gerenciar talhões", "🧪 Pesquisa experimental", "📷 Análise por foto", "📒 Diário agronômico"], index=0)


# --- MVP 0.4: desenho, gestão e levantamento GPS de talhões ---
TALHOES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "talhoes_agroNDVI.json")

def save_talhoes_catalog(data):
    """Create a backup and write catalog atomically."""
    temp_path = TALHOES_FILE + ".tmp"
    backup_path = TALHOES_FILE + ".bak"
    if os.path.exists(TALHOES_FILE):
        try:
            dated_backup = TALHOES_FILE + ".backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy2(TALHOES_FILE, dated_backup)
            shutil.copy2(TALHOES_FILE, backup_path)
        except Exception:
            pass
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, TALHOES_FILE)

NDVI_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historico_ndvi_agroNDVI.csv")

def save_ndvi_history(rows):
    """Append results to local history, de-duplicating field/date/scene."""
    if not rows:
        return 0
    incoming = pd.DataFrame(rows)
    if os.path.exists(NDVI_HISTORY_FILE):
        try:
            existing = pd.read_csv(NDVI_HISTORY_FILE)
            combined = pd.concat([existing, incoming], ignore_index=True)
        except Exception:
            combined = incoming
    else:
        combined = incoming
    keys = [k for k in ["Talhão", "Data", "ID da cena"] if k in combined.columns]
    if keys:
        combined = combined.drop_duplicates(subset=keys, keep="last")
    temp = NDVI_HISTORY_FILE + ".tmp"
    if os.path.exists(NDVI_HISTORY_FILE):
        try:
            shutil.copy2(NDVI_HISTORY_FILE, NDVI_HISTORY_FILE + ".bak")
        except Exception:
            pass
    combined.to_csv(temp, index=False, encoding="utf-8-sig")
    os.replace(temp, NDVI_HISTORY_FILE)
    return len(combined)


def load_ndvi_history():
    if not os.path.exists(NDVI_HISTORY_FILE):
        return pd.DataFrame()
    try:
        return pd.read_csv(NDVI_HISTORY_FILE)
    except Exception:
        return pd.DataFrame()

EXPERIMENTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experimentos_agroNDVI.json")

FIELD_JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "diario_agronomico_agroNDVI.json")
PHOTO_ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fotos_agroNDVI")

def load_field_journal():
    try:
        if os.path.exists(FIELD_JOURNAL_FILE):
            with open(FIELD_JOURNAL_FILE, "r", encoding="utf-8") as f:
                data=json.load(f)
            return data if isinstance(data,list) else []
    except Exception:
        return []
    return []

def save_field_journal(records):
    temp=FIELD_JOURNAL_FILE+".tmp"
    if os.path.exists(FIELD_JOURNAL_FILE):
        try: shutil.copy2(FIELD_JOURNAL_FILE,FIELD_JOURNAL_FILE+".bak")
        except Exception: pass
    with open(temp,"w",encoding="utf-8") as f: json.dump(records,f,ensure_ascii=False,indent=2)
    os.replace(temp,FIELD_JOURNAL_FILE)

def create_trial_layout(treatments, repetitions, design, seed=None):
    """Build randomized treatment allocation for DIC or DBC, one row per experimental plot."""
    rng=np.random.default_rng(seed)
    rows=[]
    treatments=[str(t).strip() for t in treatments if str(t).strip()]
    if design.startswith("DBC"):
        for block in range(1,int(repetitions)+1):
            allocation=rng.permutation(treatments)
            for pos,tr in enumerate(allocation,1):
                rows.append({"Parcela":f"B{block}-P{pos}","Tratamento":tr,"Repetição":block,"Bloco":block,"Posição no bloco":pos})
    else:
        allocation=rng.permutation([(tr,rep) for tr in treatments for rep in range(1,int(repetitions)+1)])
        for pos,(tr,rep) in enumerate(allocation,1):
            rows.append({"Parcela":f"P{pos}","Tratamento":tr,"Repetição":rep,"Bloco":1,"Posição":pos})
    return pd.DataFrame(rows)


def load_experiments():
    try:
        with open(EXPERIMENTS_FILE, "r", encoding="utf-8") as f:
            data=json.load(f)
        return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_experiments(data):
    tmp=EXPERIMENTS_FILE+".tmp"
    if os.path.exists(EXPERIMENTS_FILE):
        try: shutil.copy2(EXPERIMENTS_FILE, EXPERIMENTS_FILE+".bak")
        except Exception: pass
    with open(tmp,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
    os.replace(tmp,EXPERIMENTS_FILE)

def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(name)).strip("_") or "AgroNDVI"

def load_talhoes_catalog():
    try:
        with open(TALHOES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            return []
        # Remove duplicate saves with identical name + geometry, keeping the newest record.
        unique = {}
        for item in data:
            try:
                geom = item.get("geojson", {}).get("geometry", {})
                signature = json.dumps(geom, sort_keys=True, separators=(",", ":"))
                key = (str(item.get("name", "Talhão")).strip().lower(), signature)
                previous = unique.get(key)
                if previous is None or str(item.get("saved_at", "")) > str(previous.get("saved_at", "")):
                    unique[key] = item
            except Exception:
                unique[("invalid", str(len(unique)))] = item
        cleaned = list(unique.values())
        if len(cleaned) != len(data):
            try:
                save_talhoes_catalog(cleaned)
            except Exception:
                pass
            st.info(f"Catálogo organizado automaticamente: {len(data)-len(cleaned)} registro(s) duplicado(s) removido(s).")
        return cleaned
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def make_base_map(location, zoom_start=13, default_satellite=True):
    """Map with satellite imagery and alternate basemaps; satellite is a mosaic, not live imagery."""
    import folium
    m = folium.Map(location=location, zoom_start=zoom_start, control_scale=True, tiles=None)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri — Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        name="Satélite (mosaico)", overlay=False, control=True, show=default_satellite,
    ).add_to(m)
    folium.TileLayer("OpenStreetMap", name="Mapa de ruas", overlay=False, control=True, show=not default_satellite).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
        attr="Tiles © Esri",
        name="Referências / ruas", overlay=True, control=True, show=False,
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m

def geodesic_area_ha(geometry):
    from pyproj import Geod
    geod = Geod(ellps="WGS84")
    area_m2, _ = geod.geometry_area_perimeter(geometry)
    return abs(area_m2) / 10000

# Dashboard and report tools
NDVI_HISTORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "historico_ndvi_agroNDVI.csv")

def dashboard_catalog():
    try:
        with open(TALHOES_FILE, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, list) else []
    except Exception:
        return []

def dashboard_history():
    try:
        return pd.read_csv(NDVI_HISTORY_FILE) if os.path.exists(NDVI_HISTORY_FILE) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def build_report_pdf(df, label):
    from matplotlib.backends.backend_pdf import PdfPages
    buf = io.BytesIO()
    data = df.copy()
    if "Data" in data.columns:
        data["Data"] = pd.to_datetime(data["Data"], errors="coerce")
        data = data.dropna(subset=["Data"]).sort_values("Data")
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(11.7, 8.3))
        fig.suptitle("AgroNDVI | Relatório de monitoramento", fontsize=17, fontweight="bold")
        fig.text(0.06, 0.92, f"Área: {label} · Gerado: {datetime.now():%d/%m/%Y %H:%M}", fontsize=10)
        if not data.empty and "NDVI médio" in data.columns:
            ax = fig.add_axes([0.10, 0.50, 0.82, 0.32])
            ax.plot(data["Data"], pd.to_numeric(data["NDVI médio"], errors="coerce"), marker="o", label="NDVI médio")
            if "NDVI mediano" in data.columns:
                ax.plot(data["Data"], pd.to_numeric(data["NDVI mediano"], errors="coerce"), marker="s", label="NDVI mediano")
            ax.set_ylim(-1, 1); ax.set_ylabel("NDVI"); ax.set_xlabel("Data da imagem"); ax.grid(True, alpha=.25); ax.legend()
            avg = pd.to_numeric(data["NDVI médio"], errors="coerce")
            fig.text(.10, .42, f"Registros: {len(data)} · Inicial: {avg.iloc[0]:.3f} · Último: {avg.iloc[-1]:.3f} · Variação: {avg.iloc[-1]-avg.iloc[0]:+.3f}", fontsize=10)
            cols=[c for c in ["Data", "NDVI médio", "NDVI mediano", "NDVI mínimo", "NDVI máximo", "Pixels válidos", "Nuvens na cena (%)"] if c in data.columns]
            table_data=data[cols].tail(20).copy()
            if "Data" in table_data.columns: table_data["Data"]=table_data["Data"].dt.strftime("%Y-%m-%d")
            at=fig.add_axes([.06,.07,.88,.28]); at.axis("off")
            tab=at.table(cellText=table_data.round(3).values, colLabels=table_data.columns, cellLoc="center", loc="center")
            tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1,1.25)
        else:
            fig.text(.10,.55,"Não há dados NDVI suficientes para relatório gráfico.",fontsize=12)
        fig.text(.05,.025,"NDVI é indicador de vigor vegetativo; interprete junto de estádio, cultivar, manejo, clima e qualidade da imagem.",fontsize=8)
        pdf.savefig(fig,bbox_inches="tight"); plt.close(fig)
    buf.seek(0); return buf.getvalue()


# --- Análise visual de fotos RGB capturadas/enviadas pelo celular ---

# --- Diário agronômico local / offline ---
if app_page == "📒 Diário agronômico":
    st.title("📒 Diário agronômico de campo")
    st.caption("Registro local: funciona sem conexão para cadastrar e consultar entradas já salvas.")
    journal=load_field_journal()
    with st.form("journal_entry_form"):
        jdate=st.date_input("Data",value=date.today())
        jfield=st.text_input("Talhão / área / experimento")
        jcrop=st.selectbox("Cultura",["Soja","Milho","Algodão","Feijão","Amendoim","Outra"])
        jstage=st.text_input("Estádio fenológico (ex.: V4, R1, R5.2)")
        jtype=st.selectbox("Tipo de registro",["Monitoramento","Praga","Doença","Planta daninha","Aplicação","Adubação","Plantio","Emergência","Avaliação experimental","Colheita","Clima","Outro"])
        jseverity=st.selectbox("Intensidade / situação",["Não informado","Baixa","Moderada","Alta","Muito alta","Sem ocorrência"])
        jnotes=st.text_area("Descrição / avaliação / recomendação")
        jlat=st.text_input("Latitude (opcional)")
        jlon=st.text_input("Longitude (opcional)")
        jphoto=st.file_uploader("Foto associada (opcional)",type=["jpg","jpeg","png"],key="journal_photo")
        save_journal=st.form_submit_button("💾 Salvar registro")
    if save_journal:
        entry={"id":datetime.now().strftime("%Y%m%d%H%M%S%f"),"data":jdate.isoformat(),"talhao":jfield,"cultura":jcrop,
               "estadio":jstage,"tipo":jtype,"intensidade":jseverity,"descricao":jnotes,"latitude":jlat,"longitude":jlon,"foto":""}
        if jphoto:
            os.makedirs(PHOTO_ARCHIVE_DIR,exist_ok=True)
            ext=os.path.splitext(jphoto.name)[1].lower() or ".jpg"
            fname=safe_filename(entry["id"]+"_"+jphoto.name)
            photo_path=os.path.join(PHOTO_ARCHIVE_DIR,fname)
            with open(photo_path,"wb") as pf: pf.write(jphoto.getvalue())
            entry["foto"]=os.path.relpath(photo_path,os.path.dirname(os.path.abspath(__file__)))
        journal.append(entry)
        try:
            save_field_journal(journal); st.success("Registro salvo localmente."); st.rerun()
        except Exception as exc: st.error(f"Erro ao salvar diário: {exc}")
    st.divider()
    st.subheader("📚 Histórico")
    if journal:
        jdf=pd.DataFrame(journal)
        jdf["data"]=pd.to_datetime(jdf["data"],errors="coerce")
        cols=jdf.columns.tolist()
        filter_fields=sorted([x for x in jdf["talhao"].dropna().astype(str).unique() if x])
        selected_field=st.selectbox("Filtrar talhão",["Todos"]+filter_fields)
        if selected_field!="Todos": jdf=jdf[jdf["talhao"].astype(str)==selected_field]
        jdf=jdf.sort_values("data",ascending=False)
        st.dataframe(jdf.drop(columns=["foto"],errors="ignore"),use_container_width=True,hide_index=True)
        for _,entry in jdf.head(20).iterrows():
            with st.expander(f"{entry.get('data','')} · {entry.get('talhao','')} · {entry.get('tipo','')}"):
                st.write(entry.get("descricao",""))
                if entry.get("foto"):
                    fp=os.path.join(os.path.dirname(os.path.abspath(__file__)),str(entry["foto"]))
                    if os.path.exists(fp): st.image(fp,caption="Foto do registro",use_container_width=True)
                if entry.get("latitude") and entry.get("longitude"):
                    try: st.map(pd.DataFrame({"lat":[float(entry["latitude"])],"lon":[float(entry["longitude"])]}))
                    except Exception: st.caption("Coordenadas informadas não válidas.")
        st.download_button("⬇️ Exportar diário CSV",jdf.drop(columns=["foto"],errors="ignore").to_csv(index=False).encode("utf-8-sig"),"diario_agronomico.csv","text/csv")
        journal_json=json.dumps(journal,ensure_ascii=False,indent=2).encode("utf-8")
        st.download_button("⬇️ Exportar diário JSON (com referências de fotos)",journal_json,"diario_agronomico.json","application/json")
        if st.button("🗑️ Limpar diário (gera backup antes)",type="secondary"):
            save_field_journal([]); st.warning("Diário limpo. O backup anterior permanece."); st.rerun()
    else: st.info("Ainda não há registros.")
    st.stop()


if app_page == "📷 Análise por foto":
    st.title("📷 Análise de vegetação por fotografia")
    st.warning("Esta análise usa RGB de câmera comum. Não é NDVI verdadeiro: celulares convencionais não capturam NIR. Os índices abaixo são estimativas visuais para acompanhamento.")
    st.markdown("Tire a foto diretamente pelo celular ou envie imagens. Para comparar, mantenha distância, ângulo, iluminação e enquadramento semelhantes.")
    tab_cam, tab_upload = st.tabs(["📸 Tirar foto", "📂 Enviar fotos"])
    with tab_cam:
        camera_image = st.camera_input("Fotografar a cultura / folha / dossel", key="rgb_camera")
    with tab_upload:
        uploaded_images = st.file_uploader("Envie uma ou mais fotos", type=["jpg","jpeg","png"], accept_multiple_files=True, key="rgb_uploads")
    image_items = []
    if camera_image is not None:
        image_items.append(("Foto da câmera", camera_image))
    if uploaded_images:
        image_items.extend([(f.name, f) for f in uploaded_images])
    photo_field = st.text_input("Talhão / parcela / tratamento", key="photo_field")
    photo_note = st.text_input("Observações", key="photo_note")

    st.divider()
    st.subheader("🪴 Análise de raiz (imagem)")
    st.caption("Estimativas de área e comprimento aparente; resultados dependem de contraste, fundo, escala e raízes sobrepostas. Não substitui análise laboratorial.")
    root_img=st.file_uploader("Imagem de raiz lavada sobre fundo contrastante",type=["jpg","jpeg","png"],key="root_photo")
    scale_mm=st.number_input("Escala conhecida (mm por pixel) — opcional",min_value=0.0,value=0.0,step=0.01,help="Calibre usando régua/objeto de tamanho conhecido na mesma imagem. Sem escala, medidas ficam em pixels.")
    root_threshold=st.slider("Sensibilidade da segmentação (contraste)",0,255,95,key="root_threshold")
    if root_img:
        try:
            from PIL import Image
            from scipy import ndimage
            root_image=Image.open(root_img).convert("RGB")
            root_arr=np.asarray(root_image)
            # Roots assumed darker/brown than light background; choose contrast with background luminance.
            lum=(0.299*root_arr[:,:,0]+0.587*root_arr[:,:,1]+0.114*root_arr[:,:,2])
            root_mask=lum<root_threshold
            root_mask=ndimage.binary_opening(root_mask,iterations=1)
            root_mask=ndimage.binary_closing(root_mask,iterations=2)
            root_mask=ndimage.binary_fill_holes(root_mask)
            root_mask=ndimage.binary_opening(root_mask,iterations=1)
            # Restrict connected components to discard tiny specks.
            lab,nlab=ndimage.label(root_mask)
            sizes=ndimage.sum(root_mask,lab,range(1,nlab+1))
            keep=np.zeros_like(root_mask)
            for label,size in enumerate(sizes,1):
                if size>=max(8,root_mask.size*0.00002): keep|=(lab==label)
            area_px=int(keep.sum())
            coords=np.argwhere(keep)
            if len(coords):
                y0,x0=coords.min(axis=0); y1,x1=coords.max(axis=0)
                bbox_length_px=float(np.hypot(y1-y0,x1-x0))
                # Approximate skeleton using iterative erosion-based morphological skeleton.
                skel=np.zeros_like(keep)
                eroded=keep.copy()
                while eroded.any():
                    opened=ndimage.binary_opening(eroded)
                    skel |= eroded & ~opened
                    eroded=ndimage.binary_erosion(eroded)
                    if np.count_nonzero(eroded)<1: break
                endpoints=0
                for yy,xx in np.argwhere(skel):
                    neighborhood=skel[max(0,yy-1):yy+2,max(0,xx-1):xx+2]
                    if neighborhood.sum()-1==1: endpoints+=1
                root_length_px=int(skel.sum())
                if scale_mm>0:
                    area_cm2=area_px*(scale_mm**2)/100
                    length_cm=root_length_px*scale_mm/10
                    bbox_cm=bbox_length_px*scale_mm/10
                    st.metric("Área projetada estimada",f"{area_cm2:.2f} cm²")
                    st.metric("Comprimento de esqueleto estimado",f"{length_cm:.1f} cm")
                    st.metric("Diagonal do contorno",f"{bbox_cm:.1f} cm")
                else:
                    st.metric("Área segmentada",f"{area_px:,} pixels")
                    st.metric("Comprimento do esqueleto",f"{root_length_px:,} pixels")
                    st.metric("Diagonal do contorno",f"{bbox_length_px:.0f} pixels")
                st.metric("Extremidades do esqueleto (proxy de ramificações)",endpoints)
                overlay=root_arr.copy()
                overlay[keep]=[255,60,60]
                st.image(overlay,caption="Segmentação de raiz (sobreposição vermelha)",use_container_width=True)
                root_csv=pd.DataFrame([{"Area_pixels":area_px,"Comprimento_esqueleto_pixels":root_length_px,"Diagonal_contorno_pixels":bbox_length_px,"Extremidades_proxy":endpoints,"mm_por_pixel":scale_mm,"Area_cm2_estimada":area_px*(scale_mm**2)/100 if scale_mm>0 else None,"Comprimento_cm_estimado":root_length_px*scale_mm/10 if scale_mm>0 else None}])
                st.download_button("Exportar métricas de raiz CSV",root_csv.to_csv(index=False).encode("utf-8-sig"),"analise_raiz_AgroNDVI.csv","text/csv")
            else: st.warning("Nenhuma região de raiz identificada. Ajuste a sensibilidade ou use fundo contrastante.")
        except Exception as exc: st.error(f"Falha na análise da raiz: {exc}")

    if image_items:
        results = []
        for idx, (name, file_obj) in enumerate(image_items):
            try:
                from PIL import Image
                img = Image.open(file_obj).convert("RGB")
                arr = np.asarray(img).astype(np.float32)
                R, G, B = arr[:,:,0], arr[:,:,1], arr[:,:,2]
                denom_vari = G + R - B
                vari = np.divide(G-R, denom_vari, out=np.zeros_like(G), where=np.abs(denom_vari)>1e-6)
                exg = 2*G-R-B
                green_mask = (G > R*1.08) & (G > B*1.05) & (G > 45) & ((G-R) > 12)
                yellow_mask = (R > 70) & (G > 65) & (B < G*0.88) & (R >= G*0.72) & (R <= G*1.25) & (G > B*1.2)
                brown_mask = (R > G*1.12) & (G > B*1.15) & (R < 180) & (G < 150)
                total = max(1, arr.shape[0]*arr.shape[1])
                green_pct = green_mask.sum()/total*100
                yellow_pct = yellow_mask.sum()/total*100
                brown_pct = brown_mask.sum()/total*100
                vari_mean = float(np.mean(vari))
                exg_mean = float(np.mean(exg))
                results.append({"Imagem":name,"Cobertura verde estimada (%)":green_pct,
                                "Pixels amarelos (%)":yellow_pct,"Pixels marrons (%)":brown_pct,
                                "VARI médio (RGB)":vari_mean,"ExG médio":exg_mean})
                st.subheader(f"Imagem {idx+1}: {name}")
                st.image(img, caption=f"{img.width} × {img.height} pixels", use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("Cobertura verde estimada", f"{green_pct:.1f}%")
                c2.metric("Pixels amarelos", f"{yellow_pct:.1f}%")
                c3.metric("Pixels marrons", f"{brown_pct:.1f}%")
                c4.metric("VARI médio", f"{vari_mean:.3f}")
                # Máscara interpretativa colorida: verde classificado, amarelo, marrom, outros.
                classified = np.zeros_like(arr, dtype=np.uint8)
                classified[green_mask] = [30,190,60]
                classified[yellow_mask & ~green_mask] = [255,220,30]
                classified[brown_mask & ~green_mask & ~yellow_mask] = [150,85,40]
                st.image(classified, caption="Máscara RGB de classificação visual (verde/amarelo/marrom)", use_container_width=True)
                fig, ax = plt.subplots(figsize=(8,3.5))
                bars=["Verde","Amarelo","Marrom"]
                values=[green_pct,yellow_pct,brown_pct]
                ax.bar(bars,values)
                ax.set_ylabel("% da imagem")
                ax.set_title("Composição visual estimada")
                ax.grid(axis="y",alpha=.25)
                fig.tight_layout()
                st.pyplot(fig,use_container_width=True)
                png=io.BytesIO()
                fig.savefig(png,format="png",dpi=200,bbox_inches="tight")
                png.seek(0)
                st.download_button("Baixar gráfico PNG",png.getvalue(),file_name=f"analise_foto_{idx+1}.png",mime="image/png",key=f"rgb_plot_{idx}")
                plt.close(fig)
            except Exception as exc:
                st.error(f"Não foi possível analisar {name}: {exc}")
        if results:
            result_df=pd.DataFrame(results)
            st.divider()
            st.subheader("📊 Comparação entre fotos")
            st.dataframe(result_df.round(3),use_container_width=True,hide_index=True)
            st.download_button("⬇️ Exportar comparação CSV",result_df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="AgroNDVI_analise_fotos_RGB.csv",mime="text/csv")
            st.caption("Use fotos com condições de captura semelhantes. Cobertura verde e classificação de cores são estimativas dependentes de iluminação, câmera, sombra, fundo e enquadramento. Não diagnosticam doença nem substituem NDVI de sensor.")
    else:
        st.info("Tire uma foto usando a câmera acima ou envie imagens para começar.")


if app_page == "🏠 Painel e relatórios":
    st.header("🏠 Painel do AgroNDVI")
    cat=dashboard_catalog(); hist=dashboard_history()
    a,b,c=st.columns(3)
    a.metric("Talhões cadastrados",len(cat)); b.metric("Registros NDVI",len(hist) if not hist.empty else 0); c.metric("Histórico", "Disponível" if not hist.empty else "Sem registros")
    st.info("Acesso rápido: escolha Processar NDVI ou Desenhar/gerenciar talhões no menu lateral.")
    if cat:
        with st.expander("📋 Catálogo de talhões",expanded=True):
            cdf=pd.DataFrame([{"Nome":x.get("name","Talhão"),"Área (ha)":round(float(x.get("area_ha",0) or 0),2),"Origem":x.get("source",""),"Salvo em":str(x.get("saved_at",""))[:19]} for x in cat])
            st.dataframe(cdf,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Exportar catálogo CSV",cdf.to_csv(index=False).encode("utf-8-sig"),"catalogo_talhoes.csv","text/csv")
    if not hist.empty and "Talhão" in hist.columns:
        choices=sorted(hist["Talhão"].astype(str).dropna().unique())
        report_field=st.selectbox("Talhão para acompanhar / relatório",choices,key="dash_report_field")
        fh=hist[hist["Talhão"].astype(str)==report_field].copy()
        if "Data" in fh.columns:
            fh["Data"]=pd.to_datetime(fh["Data"],errors="coerce"); fh=fh.dropna(subset=["Data"]).sort_values("Data")
        if not fh.empty and "NDVI médio" in fh.columns:
            means=pd.to_numeric(fh["NDVI médio"],errors="coerce")
            x,y,z=st.columns(3); x.metric("NDVI mais recente",f"{means.iloc[-1]:.3f}"); y.metric("Variação",f"{means.iloc[-1]-means.iloc[0]:+.3f}"); z.metric("Cenas/datas",len(fh))
            daily=fh.groupby(fh["Data"].dt.date)["NDVI médio"].mean(); daily.index=pd.to_datetime(daily.index)
            st.line_chart(daily,y_label="NDVI médio",x_label="Data")
            st.dataframe(fh,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Exportar histórico CSV",fh.to_csv(index=False).encode("utf-8-sig"),f"historico_{report_field.replace(' ','_')}.csv","text/csv")
            st.download_button("📄 Gerar relatório PDF",build_report_pdf(fh,report_field),f"Relatorio_AgroNDVI_{report_field.replace(' ','_')}.pdf","application/pdf")
    else:
        st.warning("Sem histórico ainda. Calcule NDVI e/ou uma série temporal para habilitar gráficos e relatório.")
    st.divider(); st.subheader("🛡️ Backup e integridade")
    if os.path.exists(TALHOES_FILE):
        st.success(f"Catálogo JSON: {os.path.getsize(TALHOES_FILE):,} bytes")
        st.download_button("⬇️ Backup catálogo JSON",open(TALHOES_FILE,"rb").read(),"backup_talhoes.json","application/json")
    if os.path.exists(TALHOES_FILE+".bak"):
        st.download_button("⬇️ Backup automático anterior",open(TALHOES_FILE+".bak","rb").read(),"backup_talhoes_anterior.json","application/json")
    if os.path.exists(NDVI_HISTORY_FILE):
        st.download_button("⬇️ Backup histórico completo",open(NDVI_HISTORY_FILE,"rb").read(),"backup_historico.csv","text/csv")
    st.caption("Os arquivos são locais. Faça cópias em outro dispositivo/nuvem para proteção adicional.")
    st.subheader("🧰 Backup completo do projeto")
    backup_files=[TALHOES_FILE, NDVI_HISTORY_FILE, EXPERIMENTS_FILE, FIELD_JOURNAL_FILE]
    existing=[p for p in backup_files if os.path.exists(p)]
    if existing:
        backup_buffer=io.BytesIO()
        with zipfile.ZipFile(backup_buffer,"w",zipfile.ZIP_DEFLATED) as zf:
            for p in existing: zf.write(p,arcname=os.path.basename(p))
            if os.path.isdir(PHOTO_ARCHIVE_DIR):
                for root,dirs,files in os.walk(PHOTO_ARCHIVE_DIR):
                    for fname in files:
                        full=os.path.join(root,fname)
                        zf.write(full,arcname=os.path.relpath(full,os.path.dirname(os.path.abspath(__file__))))
        st.download_button("⬇️ Baixar backup completo (ZIP)",backup_buffer.getvalue(),"backup_completo_AgroNDVI.zip","application/zip")
    restore=st.file_uploader("Restaurar backup ZIP do AgroNDVI",type=["zip"],key="restore_backup_zip")
    if restore and st.button("⚠️ Restaurar arquivos do backup",key="restore_backup_button"):
        try:
            with zipfile.ZipFile(io.BytesIO(restore.getvalue())) as zf:
                allowed={"talhoes_agroNDVI.json","historico_ndvi_agroNDVI.csv","experimentos_agroNDVI.json","diario_agronomico_agroNDVI.json"}
                names=[n for n in zf.namelist() if os.path.basename(n) in allowed and not n.endswith("/")]
                for name in names:
                    target=os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.basename(name))
                    if os.path.exists(target): shutil.copy2(target,target+".pre_restore.bak")
                    with open(target,"wb") as out: out.write(zf.read(name))
                for name in zf.namelist():
                    if name.startswith("fotos_agroNDVI/") and not name.endswith("/"):
                        rel=os.path.normpath(name)
                        if rel.startswith("..") or os.path.isabs(rel):
                            continue
                        target=os.path.join(os.path.dirname(os.path.abspath(__file__)),rel)
                        os.makedirs(os.path.dirname(target),exist_ok=True)
                        with open(target,"wb") as out: out.write(zf.read(name))
            st.success("Backup restaurado. Arquivos anteriores foram preservados como .pre_restore.bak quando existiam.")
            st.rerun()
        except Exception as exc: st.error(f"Falha ao restaurar backup: {exc}")
    st.stop()

if app_page == "🧪 Pesquisa experimental":
    st.header("🧪 Central de ensaios experimentais")
    st.write("Cadastro, persistência, análise estatística e exportação de ensaios agronômicos. Dados são salvos localmente no computador.")

    with st.expander("🎲 Gerar croqui / randomização de tratamentos (DIC ou DBC)",expanded=False):
        design_choice=st.selectbox("Delineamento para randomização",["DIC — inteiramente casualizado","DBC — blocos casualizados"],key="layout_design")
        treatments_text=st.text_area("Tratamentos (um por linha)",value="Testemunha\nTratamento 1\nTratamento 2",key="layout_treatments")
        repetitions=st.number_input("Número de repetições / blocos",min_value=2,max_value=100,value=4,step=1,key="layout_reps")
        seed_value=st.number_input("Semente de randomização (opcional; 0 = aleatória)",min_value=0,max_value=999999999,value=0,key="layout_seed")
        if st.button("🎲 Gerar randomização",key="make_layout"):
            treatments=[x.strip() for x in treatments_text.splitlines() if x.strip()]
            if len(treatments)<2: st.error("Informe pelo menos dois tratamentos.")
            else:
                generated=create_trial_layout(treatments,repetitions,design_choice,seed_value if seed_value else None)
                st.session_state["generated_trial_layout"]=generated
        if "generated_trial_layout" in st.session_state:
            generated=st.session_state["generated_trial_layout"]
            st.dataframe(generated,use_container_width=True,hide_index=True)
            st.download_button("⬇️ Exportar croqui randomizado CSV",generated.to_csv(index=False).encode("utf-8-sig"),"croqui_randomizado_AgroNDVI.csv","text/csv",key="layout_csv_download")
            if design_choice.startswith("DBC"):
                try:
                    pivot=generated.pivot(index="Bloco",columns="Posição no bloco",values="Tratamento")
                    st.markdown("#### Visualização por bloco")
                    st.dataframe(pivot,use_container_width=True)
                except Exception: pass

    exp_catalog=load_experiments()
    action=st.radio("Modo",["➕ Novo / editar ensaio","📂 Abrir ensaio salvo","📊 Análise rápida de dados"],horizontal=True,key="exp_mode")
    if action=="📂 Abrir ensaio salvo":
        if not exp_catalog: st.info("Ainda não há ensaios salvos.")
        else:
            selected=st.selectbox("Ensaio",range(len(exp_catalog)),format_func=lambda i: f"{exp_catalog[i].get('name','Ensaio')} · {exp_catalog[i].get('updated_at','')[:10]}")
            item=exp_catalog[selected]
            st.write(f"**Projeto:** {item.get('project','')} · **Delineamento:** {item.get('design','')}")
            df=pd.DataFrame(item.get('data',[]))
            st.dataframe(df,use_container_width=True,hide_index=True)
            if not df.empty: st.download_button("⬇️ Exportar ensaio CSV",df.to_csv(index=False).encode("utf-8-sig"),safe_filename(item.get('name'))+".csv","text/csv")
            c1,c2=st.columns(2)
            with c1:
                if st.button("✏️ Carregar para edição",key="exp_load_edit"):
                    st.session_state["exp_loaded_item"]=item; st.session_state["exp_mode"]="➕ Novo / editar ensaio"; st.rerun()
            with c2:
                if st.button("🗑️ Excluir ensaio",key="exp_delete"):
                    exp_catalog.pop(selected); save_experiments(exp_catalog); st.success("Ensaio excluído."); st.rerun()
    elif action=="➕ Novo / editar ensaio":
        loaded=st.session_state.get("exp_loaded_item",{})
        with st.form("experiment_metadata_form"):
            name=st.text_input("Nome do ensaio",value=loaded.get("name","Ensaio NDVI"))
            project=st.text_input("Fazenda / projeto / safra",value=loaded.get("project",""))
            crop=st.text_input("Cultura",value=loaded.get("crop","Soja"))
            cultivar=st.text_input("Cultivar / híbrido",value=loaded.get("cultivar",""))
            location=st.text_input("Local",value=loaded.get("location",""))
            design=st.selectbox("Delineamento",["DIC — inteiramente casualizado","DBC — blocos casualizados","Fatorial (exploratório)","Outro / descritivo"],index=["DIC — inteiramente casualizado","DBC — blocos casualizados","Fatorial (exploratório)","Outro / descritivo"].index(loaded.get("design","DIC — inteiramente casualizado")) if loaded.get("design") in ["DIC — inteiramente casualizado","DBC — blocos casualizados","Fatorial (exploratório)","Outro / descritivo"] else 0)
            objective=st.text_area("Objetivo / observações",value=loaded.get("objective",""))
            save_meta=st.form_submit_button("Confirmar informações")
        if save_meta:
            st.session_state["exp_meta"]={"name":name,"project":project,"crop":crop,"cultivar":cultivar,"location":location,"design":design,"objective":objective}
        meta=st.session_state.get("exp_meta",loaded or {"name":name,"project":project,"crop":crop,"cultivar":cultivar,"location":location,"design":design,"objective":objective})
        template=pd.DataFrame({"Parcela":["P1","P2","P3","P4"],"Tratamento":["Testemunha","Testemunha","Produto A","Produto A"],"Repetição":[1,2,1,2],"Bloco":[1,2,1,2],"NDVI":[0.42,0.45,0.51,0.53]})
        saved_df=pd.DataFrame(loaded.get("data",[])) if loaded else template
        if saved_df.empty: saved_df=template
        st.download_button("⬇️ Baixar modelo CSV",template.to_csv(index=False).encode("utf-8-sig"),"modelo_ensaio_AgroNDVI.csv","text/csv")
        upload=st.file_uploader("Importar dados CSV",type=["csv"],key="exp_csv_upload")
        if upload:
            try:
                saved_df=pd.read_csv(upload); saved_df.columns=[str(c).strip() for c in saved_df.columns]
            except Exception as e: st.error(f"Erro ao importar: {e}")
        st.caption("Colunas recomendadas: Parcela, Tratamento, Repetição, Bloco e NDVI. Uma linha por parcela.")
        edited=st.data_editor(saved_df,num_rows="dynamic",use_container_width=True,key="exp_data_editor")
        required=["Parcela","Tratamento","Repetição","NDVI"]
        if all(c in edited.columns for c in required):
            data=edited.copy(); data["NDVI"]=pd.to_numeric(data["NDVI"],errors="coerce"); data=data.dropna(subset=["Tratamento","NDVI"]); data=data[(data.NDVI>=-1)&(data.NDVI<=1)]
            if not data.empty:
                summary=data.groupby("Tratamento")["NDVI"].agg(N="count",Média="mean",Desvio_padrão="std",Mínimo="min",Máximo="max").reset_index()
                summary["CV (%)"]=summary["Desvio_padrão"]/summary["Média"]*100
                st.subheader("Resumo por tratamento"); st.dataframe(summary.round(4),use_container_width=True,hide_index=True)
                st.subheader("📈 Comparação visual dos tratamentos")
                try:
                    mean_fig, mean_plot_summary = make_treatment_mean_image(data, title="NDVI médio por tratamento · barras = erro padrão")
                    st.pyplot(mean_fig, use_container_width=True)
                    mean_png=io.BytesIO(); mean_fig.savefig(mean_png,format="png",dpi=220,bbox_inches="tight"); mean_png.seek(0)
                    st.download_button("🖼️ Baixar gráfico de médias (PNG)",mean_png.getvalue(),safe_filename(meta.get("name","ensaio"))+"_medias_NDVI.png","image/png",key="exp_means_png")
                    plt.close(mean_fig)
                    fig,ax=plt.subplots(figsize=(10,5)); tr_order=list(data["Tratamento"].dropna().unique()); groups=[data.loc[data.Tratamento==tr,"NDVI"].values for tr in tr_order]
                    ax.boxplot(groups,tick_labels=tr_order,showmeans=True); ax.set_ylabel("NDVI"); ax.set_title("Distribuição de NDVI por tratamento"); ax.grid(axis="y",alpha=.25); fig.tight_layout(); st.pyplot(fig,use_container_width=True)
                    box_png=io.BytesIO(); fig.savefig(box_png,format="png",dpi=220,bbox_inches="tight"); box_png.seek(0)
                    st.download_button("🖼️ Baixar boxplot (PNG)",box_png.getvalue(),safe_filename(meta.get("name","ensaio"))+"_boxplot_NDVI.png","image/png",key="exp_box_png")
                    plt.close(fig)
                except Exception as e: st.caption(f"Gráficos comparativos indisponíveis: {e}")
                st.subheader("Análise estatística")
                design_type="DIC" if meta.get("design","").startswith("DIC") else "DBC" if meta.get("design","").startswith("DBC") else "Exploratória"
                if design_type in ("DIC","DBC") and len(data.Tratamento.unique())>=2:
                    try:
                        import statsmodels.api as sm
                        from statsmodels.formula.api import ols
                        if design_type=="DBC" and "Bloco" in data.columns and data["Bloco"].notna().sum()>0:
                            model=ols("NDVI ~ C(Tratamento) + C(Bloco)",data=data.dropna(subset=["Bloco"])).fit(); anova=sm.stats.anova_lm(model,typ=2)
                            st.write("**ANOVA com efeito de tratamento e bloco (DBC)**")
                        else:
                            model=ols("NDVI ~ C(Tratamento)",data=data).fit(); anova=sm.stats.anova_lm(model,typ=2)
                            if design_type=="DBC": st.warning("Coluna Bloco ausente/incompleta: calculado modelo sem blocos, como análise exploratória.")
                            st.write("**ANOVA de uma via (DIC / exploratória)**")
                        st.dataframe(anova.round(5),use_container_width=True)
                        term=[x for x in anova.index if "Tratamento" in x]
                        if term:
                            p=float(anova.loc[term[0],"PR(>F)"]); st.metric("p-valor do tratamento",f"{p:.5f}")
                            if p<0.05: st.success("Efeito de tratamento detectado (α = 5%).")
                            else: st.info("Não foi detectado efeito significativo de tratamento (α = 5%).")
                        residual=model.resid; mse=float(np.mean(residual**2)); mean=float(data.NDVI.mean()); cv=(np.sqrt(mse)/mean*100) if mean else np.nan
                        st.metric("CV experimental estimado (%)",f"{cv:.2f}" if np.isfinite(cv) else "n/d")
                        if len(data.Tratamento.unique())>2:
                            from statsmodels.stats.multicomp import pairwise_tukeyhsd
                            tukey=pairwise_tukeyhsd(data.NDVI,data.Tratamento,alpha=0.05)
                            st.write("**Comparação múltipla de Tukey (5%)**"); st.text(str(tukey))
                    except Exception as e: st.warning(f"Análise estatística não concluída: {e}. Confira delineamento, repetições e blocos.")
                else: st.info("Para ANOVA, informe ao menos dois tratamentos e dados replicados. O modo fatorial ainda é exploratório nesta versão.")
                st.caption("A análise pressupõe independência, resíduos aproximadamente normais e homogeneidade de variâncias. Em ensaios agronômicos, valide o delineamento, unidade experimental, blocos, transformações e diagnóstico de resíduos antes de concluir.")
                st.download_button("⬇️ CSV dos dados",data.to_csv(index=False).encode("utf-8-sig"),safe_filename(meta.get("name","ensaio"))+"_dados.csv","text/csv")
                st.download_button("⬇️ CSV do resumo",summary.to_csv(index=False).encode("utf-8-sig"),safe_filename(meta.get("name","ensaio"))+"_resumo.csv","text/csv")
                if st.button("💾 Salvar ensaio no AgroNDVI",type="primary",key="exp_save"):
                    record={**meta,"data":data.to_dict(orient="records"),"updated_at":datetime.now().isoformat(timespec="seconds")}
                    match=next((i for i,x in enumerate(exp_catalog) if x.get("name")==record["name"]),None)
                    if match is None: exp_catalog.append(record)
                    else: exp_catalog[match]=record
                    try: save_experiments(exp_catalog); st.success("Ensaio salvo localmente e com backup anterior.")
                    except Exception as e: st.error(f"Não foi possível salvar: {e}")
            else: st.info("Insira/importa dados válidos; NDVI precisa estar entre -1 e 1.")
        else: st.error("Tabela precisa conter: Parcela, Tratamento, Repetição e NDVI.")
    else:
        st.subheader("Análise rápida (sem salvar projeto)")
        quick_template=pd.DataFrame({"Parcela":["P1","P2","P3","P4"],"Tratamento":["T1","T1","T2","T2"],"Repetição":[1,2,1,2],"NDVI":[0.4,0.42,0.5,0.51]})
        qupload=st.file_uploader("CSV",type=["csv"],key="quick_exp_upload")
        qdata=pd.read_csv(qupload) if qupload else quick_template
        qedit=st.data_editor(qdata,num_rows="dynamic",key="quick_exp_editor",use_container_width=True)
        st.caption("Use o modo Novo/editar para salvar o ensaio e gerar análise estatística completa.")
        if "Tratamento" in qedit and "NDVI" in qedit:
            qedit["NDVI"]=pd.to_numeric(qedit["NDVI"],errors="coerce"); qsum=qedit.dropna(subset=["NDVI"]).groupby("Tratamento")["NDVI"].agg(N="count",Média="mean",DP="std").reset_index(); st.dataframe(qsum.round(4),use_container_width=True,hide_index=True)
    st.stop()

if app_page == "🗺️ Desenhar e gerenciar talhões":
    st.header("🗺️ Desenho e gestão de talhões")
    st.write("Delimite no mapa de satélite ou levante o perímetro caminhando com o GPS do celular. Confira a área e salve para reabrir depois.")
    st.info("A camada Satélite é um mosaico de imagens de referência (não ao vivo). Para imagens agrícolas recentes, use a busca Sentinel-2 na área Processar NDVI.")
    catalog = load_talhoes_catalog()

    management_mode = st.radio(
        "O que deseja fazer?",
        ["✏️ Desenhar no mapa", "📍 Levantar caminhando (GPS)", "💾 Talhões salvos"],
        horizontal=True,
        key="talhoes_management_mode",
    )

    try:
        import folium
        from folium.plugins import Draw
        from streamlit_folium import st_folium
    except ImportError:
        st.error("O mapa interativo requer folium e streamlit-folium. Instale as dependências do requirements.txt.")
        st.stop()

    # IMPORTANT: render only ONE interactive map/component per rerun.
    # Streamlit tabs execute all their contents; multiple st_folium components
    # mounted together can trigger React's removeChild error during reruns.
    if management_mode == "✏️ Desenhar no mapa":
        st.caption("Use Polygon ou Rectangle no mapa. Você pode editar ou remover desenhos antes de salvar.")
        # Location helper: city preset, saved field, manual coordinates, or browser GPS.
        default_lat, default_lon = -12.545, -55.712
        saved_center = None
        if catalog:
            try:
                from shapely.geometry import shape
                default_center = shape(catalog[0]["geojson"]["geometry"]).centroid
                saved_center = (float(default_center.y), float(default_center.x))
            except Exception:
                pass
        presets = {
            "Sorriso - centro": (-12.545, -55.712),
            "Talhão salvo (primeiro do catálogo)": saved_center or (-12.545, -55.712),
            "Coordenadas manuais": (default_lat, default_lon),
        }
        def apply_manual_location_preset():
            selected_preset = st.session_state.manual_location_preset
            preset_coords = presets[selected_preset]
            st.session_state.manual_lat = float(preset_coords[0])
            st.session_state.manual_lon = float(preset_coords[1])
            if selected_preset != "Coordenadas manuais":
                st.session_state.manual_zoom = 13

        st.selectbox("📍 Localizar mapa", list(presets.keys()), key="manual_location_preset", on_change=apply_manual_location_preset)
        preset_lat, preset_lon = presets[st.session_state.manual_location_preset]
        if "manual_lat" not in st.session_state:
            st.session_state.manual_lat = preset_lat
        if "manual_lon" not in st.session_state:
            st.session_state.manual_lon = preset_lon
        try:
            from streamlit_geolocation import streamlit_geolocation
            gps_for_map = streamlit_geolocation()
            if gps_for_map and gps_for_map.get("latitude") is not None and gps_for_map.get("longitude") is not None:
                st.caption(f"GPS do navegador disponível: {gps_for_map['latitude']:.6f}, {gps_for_map['longitude']:.6f}")
                if st.button("🎯 Centralizar mapa na minha localização", key="manual_use_gps"):
                    st.session_state.manual_lat = float(gps_for_map["latitude"])
                    st.session_state.manual_lon = float(gps_for_map["longitude"])
                    st.session_state.manual_zoom = 17
                    st.rerun()
            else:
                st.caption("Permita o GPS no navegador para habilitar o botão de localização atual.")
        except Exception:
            pass
        c1, c2, c3 = st.columns(3)
        with c1:
            map_lat = st.number_input("Latitude central", format="%.6f", key="manual_lat")
        with c2:
            map_lon = st.number_input("Longitude central", format="%.6f", key="manual_lon")
        with c3:
            map_zoom = st.slider("Zoom", 5, 18, 13, key="manual_zoom")
        draw_map = make_base_map([map_lat, map_lon], zoom_start=map_zoom)
        Draw(export=False, draw_options={"polyline": False, "rectangle": True, "polygon": True, "circle": False, "marker": False, "circlemarker": False}, edit_options={"edit": True, "remove": True}).add_to(draw_map)
        map_result = st_folium(
            draw_map, height=500, width=None, key="manual_talhoes_map",
            returned_objects=["all_drawings", "last_active_drawing"],
        )
        map_result = map_result or {}
        returned_drawings = map_result.get("all_drawings") or []
        active = map_result.get("last_active_drawing")
        if active and active not in returned_drawings:
            returned_drawings = list(returned_drawings) + [active]

        # Keep drawings in Streamlit session state across reruns/button clicks.
        if returned_drawings:
            st.session_state["manual_talhoes_drawings"] = returned_drawings
        drawings = st.session_state.get("manual_talhoes_drawings", [])

        c_capture, c_clear = st.columns(2)
        with c_capture:
            if st.button("📥 Capturar/atualizar desenhos do mapa", key="capture_manual_drawings"):
                if returned_drawings:
                    st.session_state["manual_talhoes_drawings"] = returned_drawings
                    st.success(f"{len(returned_drawings)} desenho(s) capturado(s).")
                else:
                    st.warning("O mapa ainda não retornou desenho. Faça o polígono/retângulo e aguarde alguns segundos.")
                st.rerun()
        with c_clear:
            if st.button("🧹 Limpar desenhos capturados", key="clear_manual_drawings"):
                st.session_state["manual_talhoes_drawings"] = []
                st.session_state["manual_drawing_choice"] = 0
                st.rerun()

        if drawings:
            drawing_options = list(range(len(drawings)))
            chosen_drawing = st.selectbox("Selecione o desenho", drawing_options, format_func=lambda i: f"Desenho {i+1}", key="manual_drawing_choice")
            feature = drawings[chosen_drawing]
            try:
                from shapely.geometry import shape
                geom = shape(feature.get("geometry", feature))
                if geom.is_empty or not geom.is_valid:
                    st.warning("O desenho está vazio ou inválido. Ajuste o polígono no mapa antes de salvar.")
                else:
                    area_ha = geodesic_area_ha(geom)
                    st.metric("Área calculada", f"{area_ha:,.2f} ha")
                    save_name = st.text_input("Nome do talhão", value=f"Talhão {len(catalog)+1}", key="manual_save_name")
                    if st.button("💾 SALVAR TALHÃO AGORA", type="primary", key="save_manual_talhao"):
                        entry = {
                            "name": save_name.strip() or f"Talhão {len(catalog)+1}",
                            "geojson": {"type": "Feature", "properties": {}, "geometry": geom.__geo_interface__},
                            "area_ha": area_ha,
                            "saved_at": datetime.now().isoformat(timespec="seconds"),
                            "source": "desenho_manual",
                        }
                        catalog.append(entry)
                        try:
                            save_talhoes_catalog(catalog)
                            # Verify the catalog was written and can be read back.
                            with open(TALHOES_FILE, "r", encoding="utf-8") as check_file:
                                check_catalog = json.load(check_file)
                            if not any(x.get("saved_at") == entry["saved_at"] and x.get("name") == entry["name"] for x in check_catalog):
                                raise IOError("A verificação após salvar não encontrou o talhão no catálogo.")
                            st.success(f"Salvo e verificado: {entry['name']} · {area_ha:.2f} ha. Arquivo: {TALHOES_FILE}")
                            st.session_state["manual_talhoes_drawings"] = []
                            st.rerun()
                        except Exception as save_exc:
                            st.error(f"Falha ao salvar o talhão: {save_exc}")
                    geojson_text = json.dumps({"type":"Feature","properties":{"name":save_name},"geometry":geom.__geo_interface__}, ensure_ascii=False)
                    st.download_button("⬇️ Baixar GeoJSON", geojson_text, file_name=f"{save_name or 'talhao'}.geojson", mime="application/geo+json", key="download_manual_geojson")
            except Exception as exc:
                st.error(f"Não foi possível validar/calcular a área do desenho: {exc}")
        else:
            st.warning("Nenhum desenho capturado ainda. Desenhe um polígono/retângulo usando as ferramentas do mapa, aguarde a atualização e clique em 'Capturar/atualizar desenhos do mapa'.")
            st.caption("Diagnóstico: o componente do mapa não retornou all_drawings nem last_active_drawing nesta atualização. Se continuar, mande uma foto desta tela para verificarmos a versão do streamlit-folium.")

    elif management_mode == "📍 Levantar caminhando (GPS)":
        st.markdown("**Levantamento caminhando:** obtenha sua posição GPS e registre pontos ao longo do perímetro. Para maior precisão, caminhe devagar e registre pontos nas mudanças de direção. Esta versão registra pontos quando você solicita a localização — não rastreia automaticamente em segundo plano.")
        if "agro_gps_points" not in st.session_state:
            st.session_state.agro_gps_points = []
        if "agro_gps_last" not in st.session_state:
            st.session_state.agro_gps_last = None
        try:
            from streamlit_geolocation import streamlit_geolocation
            gps_location = streamlit_geolocation()
            if gps_location and gps_location.get("latitude") is not None and gps_location.get("longitude") is not None:
                st.session_state.agro_gps_last = gps_location
                st.success(f"GPS obtido · precisão estimada: {gps_location.get('accuracy', 'n/d')} m")
                g1, g2 = st.columns(2)
                g1.metric("Latitude", f"{gps_location['latitude']:.7f}")
                g2.metric("Longitude", f"{gps_location['longitude']:.7f}")
                if st.button("➕ Registrar este ponto no perímetro", key="gps_add_point"):
                    point = [float(gps_location["longitude"]), float(gps_location["latitude"])]
                    points = st.session_state.agro_gps_points
                    if not points or point != points[-1]:
                        points.append(point)
                        st.session_state.agro_gps_points = points
                        st.success(f"Ponto {len(points)} registrado.")
                    else:
                        st.warning("Este ponto é igual ao último registrado.")
                    st.rerun()
            else:
                st.info("Toque no botão de localização acima e autorize o acesso ao GPS do navegador/celular.")
        except ImportError:
            st.warning("O GPS precisa do pacote streamlit-geolocation. Adicione `streamlit-geolocation` às dependências e instale-o.")

        gps_points = st.session_state.agro_gps_points
        st.write(f"**Pontos registrados: {len(gps_points)}**")
        if gps_points and st.button("↩️ Remover último ponto", key="gps_remove_last"):
            st.session_state.agro_gps_points = gps_points[:-1]
            st.rerun()
        gps_center = [-12.542, -55.721]
        if catalog:
            try:
                from shapely.geometry import shape
                gps_geom = shape(catalog[0]["geojson"]["geometry"]).centroid
                gps_center = [gps_geom.y, gps_geom.x]
            except Exception:
                pass
        if gps_points:
            gps_center = [gps_points[-1][1], gps_points[-1][0]]
        gps_map = make_base_map(location=gps_center, zoom_start=17)
        if gps_points:
            folium.PolyLine([(p[1], p[0]) for p in gps_points], color="blue", weight=4, tooltip="Trajeto registrado").add_to(gps_map)
            for idx, p in enumerate(gps_points):
                folium.CircleMarker([p[1], p[0]], radius=5, tooltip=f"Ponto {idx+1}").add_to(gps_map)
        st_folium(gps_map, height=450, width=None, key="gps_track_map", returned_objects=[])
        gps_name = st.text_input("Nome do talhão levantado", value=f"Talhão GPS {len(catalog)+1}", key="gps_talhao_name")
        b1, b2 = st.columns(2)
        with b1:
            if st.button("🔄 Limpar levantamento", key="gps_clear"):
                st.session_state.agro_gps_points = []
                st.rerun()
        with b2:
            if st.button("🔒 Fechar perímetro e salvar", type="primary", key="gps_save_polygon", disabled=len(gps_points)<3):
                try:
                    from shapely.geometry import Polygon, mapping
                    polygon = Polygon(gps_points)
                    if not polygon.is_valid:
                        polygon = polygon.buffer(0)
                    if polygon.is_empty or polygon.geom_type not in ("Polygon", "MultiPolygon"):
                        st.error("O perímetro registrado não formou um polígono válido. Registre pontos novamente.")
                    else:
                        area_ha = geodesic_area_ha(polygon)
                        entry = {"name": gps_name.strip() or f"Talhão GPS {len(catalog)+1}", "geojson": {"type":"Feature", "properties": {}, "geometry": mapping(polygon)}, "area_ha": area_ha, "saved_at": datetime.now().isoformat(timespec="seconds"), "source":"levantamento_gps"}
                        catalog.append(entry)
                        save_talhoes_catalog(catalog)
                        st.success(f"Talhão GPS salvo: {area_ha:.2f} ha")
                        st.session_state.agro_gps_points = []
                        st.rerun()
                except Exception as exc:
                    st.error(f"Erro ao fechar/salvar perímetro GPS: {exc}")
        if len(gps_points) >= 3:
            try:
                from shapely.geometry import Polygon
                preview_polygon = Polygon(gps_points)
                st.metric("Área provisória", f"{geodesic_area_ha(preview_polygon):,.2f} ha")
            except Exception:
                pass

    else:
        st.subheader("Talhões salvos")
        import_geo=st.file_uploader("Importar talhão GeoJSON",type=["geojson","json"],key="catalog_import_geojson")
        if import_geo and st.button("📥 Importar GeoJSON para catálogo",key="catalog_import_button"):
            try:
                imported=load_geojson_bytes(import_geo.getvalue())
                from shapely.geometry import mapping
                for idx,geom in enumerate(imported.geometry):
                    entry_name=(str(imported.iloc[idx].get("name", "Talhão importado")) if hasattr(imported.iloc[idx],"get") else "Talhão importado")
                    catalog.append({"name":entry_name if len(imported)==1 else f"{entry_name} {idx+1}","geojson":{"type":"Feature","properties":{},"geometry":mapping(geom)},"area_ha":geodesic_area_ha(geom),"saved_at":datetime.now().isoformat(timespec="seconds"),"source":"import_geojson"})
                save_talhoes_catalog(catalog); st.success(f"{len(imported)} geometria(s) importada(s)!"); st.rerun()
            except Exception as exc: st.error(f"Importação falhou: {exc}")
        if catalog:
            saved_names = [f"{x.get('name','Talhão')} · {x.get('area_ha',0):.2f} ha" for x in catalog]
            selected_saved = st.selectbox("Selecione para visualizar/exportar", range(len(catalog)), format_func=lambda i: saved_names[i], key="saved_talhao_select")
            selected = catalog[selected_saved]
            st.metric("Área", f"{selected.get('area_ha',0):,.2f} ha")
            st.caption(f"Origem: {selected.get('source','não informada')} · salvo em: {selected.get('saved_at','n/d')}")
            st.download_button("⬇️ Exportar GeoJSON", json.dumps(selected["geojson"], ensure_ascii=False), file_name=f"{selected.get('name','talhao')}.geojson", mime="application/geo+json", key="saved_geojson_download")
            if st.button("🗑️ Excluir talhão selecionado", key="saved_delete"):
                catalog.pop(selected_saved)
                save_talhoes_catalog(catalog)
                st.rerun()
            try:
                from shapely.geometry import shape
                from shapely.ops import unary_union
                all_geoms = [shape(x["geojson"]["geometry"]) for x in catalog if x.get("geojson", {}).get("geometry")]
                combined = unary_union(all_geoms)
                minx, miny, maxx, maxy = combined.bounds
                saved_map = make_base_map(location=[(miny+maxy)/2, (minx+maxx)/2], zoom_start=13)
                saved_map.fit_bounds([[miny, minx], [maxy, maxx]])
            except Exception:
                saved_map = make_base_map(location=[-12.542, -55.721], zoom_start=13)
            for item in catalog:
                folium.GeoJson(item["geojson"], name=item.get("name"), tooltip=f"{item.get('name')} · {item.get('area_ha',0):.2f} ha").add_to(saved_map)
            st_folium(saved_map, height=450, width=None, key="saved_talhoes_map", returned_objects=[])
        else:
            st.info("Ainda não há talhões salvos. Use o desenho manual ou o levantamento GPS.")
    st.stop()


def load_geojson_bytes(raw_bytes):
    import geopandas as gpd
    raw = json.loads(raw_bytes.decode("utf-8"))
    if raw.get("type") == "FeatureCollection":
        gdf = gpd.GeoDataFrame.from_features(raw["features"], crs="EPSG:4326")
    elif raw.get("type") == "Feature":
        gdf = gpd.GeoDataFrame.from_features([raw], crs="EPSG:4326")
    elif raw.get("type") in ("Polygon", "MultiPolygon"):
        from shapely.geometry import shape
        gdf = gpd.GeoDataFrame({"geometry": [shape(raw)]}, crs="EPSG:4326")
    else:
        raise ValueError("O arquivo precisa ser um GeoJSON Feature, FeatureCollection, Polygon ou MultiPolygon.")
    gdf = gdf[gdf.geometry.notnull() & ~gdf.geometry.is_empty]
    if gdf.empty:
        raise ValueError("O GeoJSON não contém geometrias válidas.")
    if not gdf.geometry.is_valid.all():
        gdf["geometry"] = gdf.geometry.buffer(0)
    return gdf

def read_uploaded_raster(upload):
    upload.seek(0)
    with MemoryFile(upload.read()) as mem:
        with mem.open() as ds:
            arr = ds.read(1).astype("float32")
            profile = ds.profile.copy()
            nodata = ds.nodata
    return arr, profile, nodata

def align_to_reference(src_arr, src_profile, ref_profile):
    if (src_arr.shape == (ref_profile["height"], ref_profile["width"]) and
        src_profile.get("transform") == ref_profile.get("transform") and
        src_profile.get("crs") == ref_profile.get("crs")):
        return src_arr
    dst = np.zeros((ref_profile["height"], ref_profile["width"]), dtype="float32")
    reproject(
        source=src_arr,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        dst_transform=ref_profile["transform"],
        dst_crs=ref_profile["crs"],
        resampling=Resampling.bilinear,
    )
    return dst

def search_sentinel_scenes(geojson, start_date, end_date, max_cloud):
    """Search public Sentinel-2 L2A scenes in Microsoft Planetary Computer STAC."""
    from pystac_client import Client
    import planetary_computer

    catalog = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=geojson,
        datetime=f"{start_date.isoformat()}T00:00:00Z/{end_date.isoformat()}T23:59:59Z",
        query={"eo:cloud_cover": {"lt": float(max_cloud) + 0.001}},
        max_items=100,
    )
    items = list(search.items())
    # Prefer newest acquisitions first; cloud percentage is the secondary ranking.
    items.sort(
        key=lambda item: (
            item.datetime or datetime.min,
            -float(item.properties.get("eo:cloud_cover", 100)),
        ),
        reverse=True,
    )
    return [item.to_dict() for item in items]

def download_sentinel_ndvi(item_dict, geojson):
    """
    Read Sentinel-2 L2A B04/B08 and SCL from signed cloud-optimized GeoTIFFs.
    Crop to AOI, align all bands to B04's 10 m grid, and mask cloud/shadow classes.
    """
    import planetary_computer
    import pystac

    item = pystac.Item.from_dict(item_dict)
    item = planetary_computer.sign(item)

    required = ["B04", "B08", "SCL"]
    missing = [name for name in required if name not in item.assets]
    if missing:
        raise RuntimeError(f"A cena selecionada não contém os ativos necessários: {', '.join(missing)}")

    # B04 is 10 m and defines the output grid.
    # The uploaded AOI is WGS84 (EPSG:4326), while Sentinel-2 assets are
    # usually in the tile's projected UTM CRS. Reproject the AOI before masking.
    from rasterio.warp import transform_geom
    with rasterio.open(item.assets["B04"].href) as red_ds:
        if red_ds.crs is None:
            raise RuntimeError("A banda B04 não informa o sistema de coordenadas (CRS).")
        try:
            aoi_in_raster_crs = transform_geom(
                "EPSG:4326", red_ds.crs, geojson, precision=8
            )
            red_img, transform = mask(
                red_ds, [aoi_in_raster_crs], crop=True, filled=False
            )
        except ValueError as exc:
            raise RuntimeError(
                "O polígono do talhão não se sobrepõe à imagem Sentinel-2 após "
                "a transformação para o CRS da cena. Confira se o GeoJSON pertence "
                "à mesma área da imagem selecionada."
            ) from exc
        red = red_img[0].astype("float32")
        out_profile = red_ds.profile.copy()
        out_profile.update(
            height=red.shape[0],
            width=red.shape[1],
            transform=transform,
            count=1,
            dtype="float32",
        )
        out_crs = red_ds.crs

    def read_to_output(asset_key, resampling):
        with rasterio.open(item.assets[asset_key].href) as src:
            destination = np.zeros(red.shape, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=destination,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=out_crs,
                dst_nodata=0,
                resampling=resampling,
            )
        return destination

    nir = read_to_output("B08", Resampling.bilinear)
    scl = read_to_output("SCL", Resampling.nearest).astype("uint8")

    red_data = np.asarray(red.filled(np.nan), dtype="float32")
    # Sentinel-2 SCL: mask no-data, saturated/defective, cloud shadow,
    # medium/high probability clouds, cirrus and snow/ice.
    bad_scl = np.isin(scl, [0, 1, 3, 8, 9, 10, 11])
    valid = (
        np.isfinite(red_data) & np.isfinite(nir) &
        (red_data > 0) & (nir > 0) & (~bad_scl)
    )
    denom = nir + red_data
    valid &= denom > 0
    ndvi = np.full(red.shape, np.nan, dtype="float32")
    ndvi[valid] = (nir[valid] - red_data[valid]) / denom[valid]
    valid &= np.isfinite(ndvi) & (ndvi >= -1) & (ndvi <= 1)
    ndvi[~valid] = np.nan

    # Preserve mask outside AOI and cloud pixels as nodata in output.
    out_profile.update(dtype="float32", nodata=-9999.0, compress="deflate")
    return ndvi, out_profile, item.properties, int(valid.sum()), int(bad_scl.sum())

def make_map_image(ndvi, valid, title="Mapa NDVI"):
    fig, ax = plt.subplots(figsize=(10, 7))
    masked = np.ma.masked_where(~valid, ndvi)
    im = ax.imshow(masked, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_title(title)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("NDVI")
    fig.tight_layout()
    return fig

def make_ndvi_zones_image(ndvi, valid, title="Zonas de vigor NDVI"):
    fig, ax = plt.subplots(figsize=(10, 7))
    labels = ["< 0", "0–0,20", "0,20–0,40", "0,40–0,60", "> 0,60"]
    cmap = plt.get_cmap("RdYlGn", len(labels))
    classified = np.digitize(ndvi, [-0.0, 0.2, 0.4, 0.6], right=False)
    masked = np.ma.masked_where(~valid, classified)
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=len(labels)-1)
    ax.set_title(title); ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, ticks=range(len(labels)), fraction=0.035, pad=0.02)
    cbar.ax.set_yticklabels(labels); cbar.set_label("Classes NDVI (faixas visuais)")
    fig.tight_layout()
    return fig

def make_treatment_mean_image(data, value_col="NDVI", treatment_col="Tratamento", title="Média por tratamento ± erro padrão"):
    summary = data.groupby(treatment_col)[value_col].agg(["mean", "std", "count"]).reset_index()
    summary["se"] = summary["std"] / np.sqrt(summary["count"].clip(lower=1))
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(summary))
    ax.bar(x, summary["mean"], yerr=summary["se"], capsize=5)
    ax.set_xticks(x, summary[treatment_col].astype(str), rotation=25, ha="right")
    ax.set_ylabel(value_col); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.tight_layout()
    return fig, summary

def make_difference_image(diff, valid, title="Diferença NDVI (data 1 − data 2)"):
    fig, ax = plt.subplots(figsize=(10, 7))
    masked = np.ma.masked_where(~valid, diff)
    lim = max(0.05, float(np.nanpercentile(np.abs(diff[valid]), 98))) if valid.any() else 0.1
    im = ax.imshow(masked, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_title(title)
    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Δ NDVI")
    fig.tight_layout()
    return fig

def make_ndvi_satellite_map(ndvi, profile, field, geojson=None):
    """Render NDVI as transparent PNG over a satellite basemap, in geographic bounds."""
    import folium
    from rasterio.transform import array_bounds
    from rasterio.warp import transform_bounds
    from PIL import Image

    valid = np.isfinite(ndvi)
    if not valid.any():
        return None
    h, w = ndvi.shape
    bounds = array_bounds(h, w, profile["transform"])
    crs = profile.get("crs")
    if crs:
        west, south, east, north = transform_bounds(crs, "EPSG:4326", *bounds, densify_pts=21)
    else:
        west, south, east, north = bounds
    cmap = plt.get_cmap("RdYlGn")
    normalized = np.clip((ndvi + 1) / 2, 0, 1)
    rgba = (cmap(normalized) * 255).astype("uint8")
    rgba[..., 3] = np.where(valid, 210, 0).astype("uint8")
    png = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(png, format="PNG")
    center = [(south+north)/2, (west+east)/2]
    m = make_base_map(center, zoom_start=15, default_satellite=True)
    folium.raster_layers.ImageOverlay(image=png.getvalue(), bounds=[[south, west], [north, east]], opacity=0.82, name="NDVI", interactive=True, cross_origin=False, zindex=2).add_to(m)
    if geojson:
        folium.GeoJson(geojson, name="Limite do talhão", style_function=lambda _: {"color":"white", "weight":2, "fillOpacity":0}).add_to(m)
    m.fit_bounds([[south, west], [north, east]])
    return m

def show_results(ndvi, profile, field, acquired, source_note, extra_note=""):
    valid = np.isfinite(ndvi)
    vals = ndvi[valid]
    if vals.size == 0:
        st.error("Não há pixels válidos após o recorte e a máscara de nuvens. Tente outra data ou confira o polígono.")
        return

    st.success(f"NDVI calculado para {field} · {acquired}")
    if extra_note:
        st.info(extra_note)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NDVI médio", f"{np.mean(vals):.3f}")
    m2.metric("NDVI mediano", f"{np.median(vals):.3f}")
    m3.metric("NDVI mínimo", f"{np.min(vals):.3f}")
    m4.metric("NDVI máximo", f"{np.max(vals):.3f}")

    tab1, tab2, tab3, tab4 = st.tabs(["Mapa NDVI", "Estatísticas", "Exportar", "NDVI no satélite"])
    with tab1:
        map_fig=make_map_image(ndvi, valid, f"AgroNDVI · {field} · {acquired}")
        st.pyplot(map_fig, use_container_width=True)
        map_png=io.BytesIO(); map_fig.savefig(map_png,format="png",dpi=240,bbox_inches="tight"); map_png.seek(0)
        st.download_button("🖼️ Baixar mapa contínuo NDVI (PNG)",map_png.getvalue(),file_name=f"AgroNDVI_{safe_filename(field)}_{acquired}_mapa.png",mime="image/png",key="ndvi_map_png")
        plt.close(map_fig)
        st.markdown("#### 🟩 Mapa por zonas de vigor")
        zone_fig=make_ndvi_zones_image(ndvi,valid,f"Zonas de vigor · {field} · {acquired}")
        st.pyplot(zone_fig,use_container_width=True)
        zone_png=io.BytesIO(); zone_fig.savefig(zone_png,format="png",dpi=240,bbox_inches="tight"); zone_png.seek(0)
        st.download_button("🖼️ Baixar mapa por zonas (PNG)",zone_png.getvalue(),file_name=f"AgroNDVI_{safe_filename(field)}_{acquired}_zonas.png",mime="image/png",key="ndvi_zones_png")
        plt.close(zone_fig)
        st.caption("Vermelho = menor NDVI; amarelo = intermediário; verde = maior NDVI. A interpretação depende da cultura, estádio, solo e condições da imagem.")
        st.caption("As faixas do mapa por zonas são classes visuais genéricas; ajuste a interpretação ao cultivo e ao estádio fenológico.")
        st.caption(source_note)
    with tab4:
        try:
            sat_map = make_ndvi_satellite_map(ndvi, profile, field)
            if sat_map is not None:
                from streamlit_folium import st_folium
                st_folium(sat_map, height=600, width=None, key=f"ndvi_sat_map_{str(acquired).replace('-', '')}", returned_objects=[])
                st.caption("NDVI sobre mosaico de satélite. A imagem-base é de referência e não representa transmissão ao vivo.")
            else:
                st.warning("Não há pixels válidos para mostrar no mapa.")
        except Exception as exc:
            st.warning(f"Não foi possível montar o mapa sobre satélite: {exc}")
    with tab2:
        # Faixas NDVI de 0,05, alinhadas à escala e limitadas a [-1, 1].
        # A borda final é ampliada quando necessário para incluir o máximo
        # observado, sem criar faixas vazias fora do intervalo dos dados.
        bin_width = 0.05
        observed_min = float(np.min(vals))
        observed_max = float(np.max(vals))
        min_edge = max(-1.0, np.floor(observed_min / bin_width) * bin_width)
        max_edge = min(1.0, np.ceil(observed_max / bin_width) * bin_width)
        if np.isclose(max_edge, min_edge):
            max_edge = min(1.0, min_edge + bin_width)
        edges = np.arange(min_edge, max_edge + bin_width / 2, bin_width, dtype=float)
        if edges.size < 2:
            edges = np.array([min_edge, min(1.0, min_edge + bin_width)], dtype=float)
        # Evita erro de ponto flutuante em limites e garante inclusão do valor máximo.
        if edges[-1] < observed_max and not np.isclose(edges[-1], observed_max):
            edges = np.append(edges, min(1.0, edges[-1] + bin_width))
        counts, edges = np.histogram(vals, bins=edges)
        hist_df = pd.DataFrame({
            "Faixa NDVI": [f"{edges[i]:.2f} a {edges[i+1]:.2f}" for i in range(len(edges)-1)],
            "Pixels": counts,
            "% dos pixels": (counts / vals.size * 100).round(2),
        })
        # Exibe somente faixas que contêm pixels para facilitar a leitura.
        hist_df = hist_df[hist_df["Pixels"] > 0].reset_index(drop=True)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)
        zone_edges=[-1.0,0.0,0.2,0.4,0.6,1.0]
        zone_labels=["< 0","0–0,20","0,20–0,40","0,40–0,60","> 0,60"]
        zone_counts=[]
        for i,label in enumerate(zone_labels):
            low,high=zone_edges[i],zone_edges[i+1]
            mask_class=(vals>=low)&((vals<high) if i<len(zone_labels)-1 else (vals<=high))
            zone_counts.append(int(mask_class.sum()))
        zone_df=pd.DataFrame({"Zona de vigor":zone_labels,"Pixels":zone_counts,"Área estimada (ha)":np.array(zone_counts)*0.01,"% da área válida":np.array(zone_counts)/max(1,vals.size)*100})
        st.markdown("#### Distribuição por zona de vigor")
        st.dataframe(zone_df.round(2),use_container_width=True,hide_index=True)
        figz,axz=plt.subplots(figsize=(8,3.5)); axz.bar(zone_df["Zona de vigor"],zone_df["% da área válida"]); axz.set_ylabel("% dos pixels válidos"); axz.set_title("Distribuição espacial do vigor"); axz.grid(axis="y",alpha=.25); figz.tight_layout(); st.pyplot(figz,use_container_width=True)
        zpng=io.BytesIO(); figz.savefig(zpng,format="png",dpi=220,bbox_inches="tight"); zpng.seek(0)
        st.download_button("🖼️ Baixar gráfico de zonas (PNG)",zpng.getvalue(),file_name=f"AgroNDVI_{safe_filename(field)}_{acquired}_distribuicao_zonas.png",mime="image/png",key="ndvi_zones_chart_png")
        plt.close(figz)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.hist(vals, bins=edges, edgecolor="black", linewidth=0.4)
        ax.set_xlabel("NDVI")
        ax.set_ylabel("Número de pixels")
        ax.set_title("Distribuição dos valores de NDVI")
        ax.set_xlim(float(edges[0]), float(edges[-1]))
        st.pyplot(fig, use_container_width=True)
        st.caption(f"Pixels válidos: {vals.size:,}. Na grade de 10 m, cada pixel representa aproximadamente 100 m².")
    with tab3:
        tif_buffer = io.BytesIO()
        with MemoryFile() as mem:
            with mem.open(**profile) as ds:
                ds.write(np.where(valid, ndvi, -9999).astype("float32"), 1)
            tif_buffer.write(mem.read())
        stats = pd.DataFrame([{
            "talhao": field,
            "data_imagem": str(acquired),
            "ndvi_medio": float(np.mean(vals)),
            "ndvi_mediano": float(np.median(vals)),
            "ndvi_minimo": float(np.min(vals)),
            "ndvi_maximo": float(np.max(vals)),
            "pixels_validos": int(vals.size),
            "area_pixel_m2_aprox": 100,
        }])
        st.download_button("Baixar NDVI (GeoTIFF)", data=tif_buffer.getvalue(), file_name="agroNDVI_resultado.tif", mime="image/tiff")
        st.download_button("Baixar estatísticas (CSV)", data=stats.to_csv(index=False).encode("utf-8-sig"), file_name="agroNDVI_estatisticas.csv", mime="text/csv")
        st.dataframe(stats, use_container_width=True, hide_index=True)

if app_page == "🌿 Processar NDVI" and source_mode == "Sentinel-2 (talhão salvo ou GeoJSON)":
    st.subheader("1. Selecione a área de análise")
    catalog = load_talhoes_catalog()
    area_source = st.radio("Origem do limite", ["Usar talhão salvo", "Enviar GeoJSON"], horizontal=True, key="sentinel_area_source")
    aoi_file = None
    selected_catalog_item = None
    intersects = None
    selected_field_label = field_name
    if area_source == "Usar talhão salvo":
        if catalog:
            labels = [f"{x.get('name','Talhão')} · {float(x.get('area_ha',0)):.2f} ha · {x.get('source','') }" for x in catalog]
            chosen = st.selectbox("Talhão cadastrado", range(len(catalog)), format_func=lambda i: labels[i], key="sentinel_saved_field")
            selected_catalog_item = catalog[chosen]
            selected_field_label = selected_catalog_item.get("name", field_name)
            try:
                intersects = selected_catalog_item["geojson"]["geometry"]
            except Exception:
                st.error("O talhão selecionado não tem geometria GeoJSON válida.")
                intersects = None
            if intersects:
                st.caption(f"Área cadastrada: {float(selected_catalog_item.get('area_ha',0)):.2f} ha. Não é necessário exportar GeoJSON.")
        else:
            st.warning("Seu catálogo ainda não tem talhões. Desenhe/salve um talhão na área 'Desenhar e gerenciar talhões' ou selecione 'Enviar GeoJSON'.")
            area_source = "Enviar GeoJSON"
    if area_source == "Enviar GeoJSON":
        aoi_file = st.file_uploader("Limite do talhão — GeoJSON", type=["geojson", "json"], key="aoi")
        if aoi_file:
            try:
                aoi_gdf = load_geojson_bytes(aoi_file.getvalue())
                aoi_features = json.loads(aoi_gdf.to_json())["features"]
                if len(aoi_features) == 1:
                    intersects = aoi_features[0]["geometry"]
                else:
                    from shapely.geometry import mapping
                    from shapely.ops import unary_union
                    intersects = mapping(unary_union(list(aoi_gdf.geometry)))
                selected_field_label = field_name
            except Exception as exc:
                st.error(f"GeoJSON inválido: {exc}")
                intersects = None
    if intersects:
        try:
            import folium
            from streamlit_folium import st_folium
            from shapely.geometry import shape
            geom = shape(intersects)
            center = [geom.centroid.y, geom.centroid.x]
            preview = make_base_map(center, zoom_start=15, default_satellite=True)
            folium.GeoJson(intersects, name="Talhão selecionado", style_function=lambda _: {"color":"#00ff88", "weight":3, "fillOpacity":0.15}, tooltip=selected_field_label).add_to(preview)
            preview.fit_bounds([[geom.bounds[1], geom.bounds[0]], [geom.bounds[3], geom.bounds[2]]])
            st_folium(preview, height=420, width=None, key="sentinel_aoi_preview", returned_objects=[])
        except Exception as exc:
            st.caption(f"Pré-visualização do mapa indisponível: {exc}")

    st.subheader("2. Período e qualidade das imagens")
    today = date.today()
    c1, c2, c3 = st.columns(3)
    with c1:
        quick_period = st.selectbox("Busca rápida", ["Personalizado", "Últimos 30 dias", "Últimos 60 dias", "Últimos 90 dias", "Últimos 180 dias"], index=2, key="quick_sentinel_period")
    with c2:
        if quick_period == "Personalizado":
            start_date = st.date_input("Data inicial", value=today - timedelta(days=60), key="start")
        else:
            days = int(quick_period.split()[1])
            start_date = today - timedelta(days=days)
    with c3:
        end_date = st.date_input("Data final", value=today, key="end")
    max_cloud = st.slider("Cobertura máxima de nuvens na cena (%)", 0, 100, 50, 5, key="max_cloud_sentinel")
    st.caption("Sentinel-2 não é vídeo ao vivo: a constelação tem revisita nominal de cerca de 5 dias, mas disponibilidade, nuvens e processamento podem atrasar ou impedir uma imagem útil.")
    if start_date > end_date:
        st.error("A data inicial não pode ser posterior à data final.")
    elif st.button("🔎 Buscar imagens Sentinel-2", type="primary", disabled=not intersects):
        try:
            with st.spinner("Consultando catálogo público Sentinel-2..."):
                scenes = search_sentinel_scenes(intersects, start_date, end_date, max_cloud)
            st.session_state["agro_scenes"] = scenes
            st.session_state.pop("agro_ndvi_timeseries", None)
            st.session_state.pop("agro_ndvi_timeseries_errors", None)
            st.session_state["agro_aoi"] = intersects
            st.session_state["agro_aoi_name"] = selected_field_label
            st.session_state["agro_aoi_geojson"] = intersects
            st.success(f"Encontradas {len(scenes)} cenas candidatas.") if scenes else st.warning("Nenhuma cena encontrada. Amplie o período ou aumente o limite de nuvens.")
        except Exception as exc:
            st.error(f"Falha na busca: {exc}")
            st.info("Confira a conexão e tente novamente. O catálogo público pode estar temporariamente indisponível.")

    scenes = st.session_state.get("agro_scenes", [])
    if scenes:
        st.subheader("3. Imagens disponíveis")
        scene_rows = []
        options = []
        for idx, scene in enumerate(scenes):
            props = scene.get("properties", {})
            acquired = props.get("datetime", "Data desconhecida")
            cloud = props.get("eo:cloud_cover", "n/d")
            tile = props.get("s2:mgrs_tile", "tile não informado")
            scene_id = scene.get("id", "")
            options.append(f"{acquired[:10]} · nuvens {float(cloud):.1f}% · {tile} · {scene_id}")
            scene_rows.append({"Data": acquired[:10], "Nuvens na cena (%)": cloud, "Tile": tile, "ID": scene_id})
        st.dataframe(pd.DataFrame(scene_rows), use_container_width=True, hide_index=True)
        selected_idx = st.selectbox("Imagem para calcular", range(len(options)), format_func=lambda x: options[x], key="sentinel_scene_select")
        compare = st.checkbox("Comparar com outra data", value=False, key="sentinel_compare_checkbox")
        compare_idx = None
        if compare:
            compare_options = [i for i in range(len(options)) if i != selected_idx]
            if compare_options:
                compare_idx = st.selectbox("Imagem de comparação", compare_options, format_func=lambda x: options[x], key="sentinel_compare_select")
            else:
                st.info("A busca precisa retornar pelo menos duas cenas para comparar datas.")
        if st.button("🌿 Calcular NDVI" + (" e comparar datas" if compare and compare_idx is not None else " da imagem selecionada"), type="primary", key="sentinel_calculate"):
            try:
                aoi = st.session_state.get("agro_aoi")
                with st.spinner("Baixando bandas Sentinel-2, alinhando pixels e mascarando nuvens/sombras..."):
                    ndvi, profile, props, valid_count, masked_count = download_sentinel_ndvi(scenes[selected_idx], aoi)
                acquired = props.get("datetime", "Data não informada")[:10]
                cloud_cover = props.get("eo:cloud_cover", "n/d")
                note = f"Sentinel-2 L2A · {scenes[selected_idx].get('id')} · nuvens na cena: {float(cloud_cover):.1f}% · máscara SCL aplicada."
                show_results(ndvi, profile, st.session_state.get("agro_aoi_name", selected_field_label), acquired, note,
                    f"Pixels válidos após máscara: {valid_count:,}. Pixels/classes descartados pela máscara SCL: {masked_count:,}.")
                if compare and compare_idx is not None:
                    with st.spinner("Processando a segunda data para comparação..."):
                        ndvi2, profile2, props2, valid2, masked2 = download_sentinel_ndvi(scenes[compare_idx], aoi)
                    date2 = props2.get("datetime", "Data não informada")[:10]
                    vals1, vals2 = ndvi[np.isfinite(ndvi)], ndvi2[np.isfinite(ndvi2)]
                    st.subheader("📈 Comparação temporal")
                    if vals1.size and vals2.size:
                        delta = float(np.mean(vals1)-np.mean(vals2))
                        x1, x2, x3 = st.columns(3)
                        x1.metric(f"NDVI médio · {acquired}", f"{np.mean(vals1):.3f}")
                        x2.metric(f"NDVI médio · {date2}", f"{np.mean(vals2):.3f}")
                        x3.metric("Diferença entre médias", f"{delta:+.3f}", help="Valor da primeira data menos a segunda.")
                        st.caption("A diferença de médias deve ser interpretada considerando estádio da cultura, cultivar, manejo, solo, qualidade atmosférica e pixels válidos em cada data.")
                        comp_df = pd.DataFrame([{"Data": acquired, "NDVI médio": float(np.mean(vals1)), "Mediana": float(np.median(vals1)), "Pixels válidos": int(vals1.size)}, {"Data": date2, "NDVI médio": float(np.mean(vals2)), "Mediana": float(np.median(vals2)), "Pixels válidos": int(vals2.size)}])
                        st.dataframe(comp_df, use_container_width=True, hide_index=True)
                        st.download_button("⬇️ Baixar comparação CSV", comp_df.to_csv(index=False).encode("utf-8-sig"), file_name="agroNDVI_comparacao_datas.csv", mime="text/csv", key="comparison_csv_download")
                        try:
                            aligned2 = align_to_reference(ndvi2, profile2, profile)
                            diff_valid = np.isfinite(ndvi) & np.isfinite(aligned2) & (aligned2 >= -1) & (aligned2 <= 1)
                            diff = np.full(ndvi.shape, np.nan, dtype="float32")
                            diff[diff_valid] = ndvi[diff_valid] - aligned2[diff_valid]
                            st.subheader("Mapa de diferença espacial")
                            st.pyplot(make_difference_image(diff, diff_valid, f"Δ NDVI · {acquired} − {date2}"), use_container_width=True)
                            st.caption("Azul/vermelho conforme a escala: valores positivos indicam NDVI maior na primeira data; negativos, menor. Use apenas pixels válidos em ambas as datas.")
                            diff_profile = profile.copy()
                            diff_profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
                            diff_buffer = io.BytesIO()
                            with MemoryFile() as mem:
                                with mem.open(**diff_profile) as ds:
                                    ds.write(np.where(diff_valid, diff, -9999).astype("float32"), 1)
                                diff_buffer.write(mem.read())
                            st.download_button("⬇️ Baixar mapa de diferença GeoTIFF", diff_buffer.getvalue(), file_name="agroNDVI_diferenca_datas.tif", mime="image/tiff", key="difference_geotiff_download")
                        except Exception as compare_exc:
                            st.warning(f"Comparação espacial não foi gerada: {compare_exc}")
                    else:
                        st.warning("Uma das imagens não tem pixels válidos suficientes para comparação.")
            except Exception as exc:
                st.error(f"Falha ao calcular/processar a cena: {exc}")
                st.info("Tente outra cena, preferencialmente com menor cobertura de nuvens. A disponibilidade depende do catálogo e da conexão.")

    # Série temporal automatizada de NDVI para todas as cenas Sentinel encontradas.
    if scenes:
        st.divider()
        st.subheader("📈 Série temporal do NDVI")
        st.write("Calcula o NDVI de todas as cenas Sentinel-2 encontradas no período e gera evolução temporal por talhão.")
        if st.button("📊 Calcular evolução NDVI de todas as datas", type="secondary", key="sentinel_timeseries_button"):
            ts_rows = []
            ts_errors = []
            progress = st.progress(0, text="Preparando processamento das cenas...")
            for i, scene in enumerate(scenes):
                scene_props = scene.get("properties", {})
                scene_date = (scene_props.get("datetime") or "")[:10]
                scene_cloud = scene_props.get("eo:cloud_cover", np.nan)
                try:
                    progress.progress((i + 1) / max(1, len(scenes)), text=f"Processando {i+1}/{len(scenes)} · {scene_date}...")
                    scene_ndvi, scene_profile, scene_result_props, scene_valid, scene_masked = download_sentinel_ndvi(
                        scene, st.session_state.get("agro_aoi")
                    )
                    vals = scene_ndvi[np.isfinite(scene_ndvi)]
                    if vals.size == 0:
                        raise RuntimeError("Sem pixels válidos após máscara de nuvens/sombras.")
                    actual_date = (scene_result_props.get("datetime") or scene_date)[:10]
                    ts_rows.append({
                        "Talhão": selected_field_label,
                        "Data": actual_date,
                        "NDVI médio": float(np.mean(vals)),
                        "NDVI mediano": float(np.median(vals)),
                        "NDVI mínimo": float(np.min(vals)),
                        "NDVI máximo": float(np.max(vals)),
                        "Desvio padrão": float(np.std(vals)),
                        "Pixels válidos": int(vals.size),
                        "Nuvens na cena (%)": float(scene_cloud) if scene_cloud is not None else np.nan,
                        "ID da cena": scene.get("id", ""),
                    })
                except Exception as series_exc:
                    ts_errors.append({"Data": scene_date, "ID da cena": scene.get("id", ""), "Erro": str(series_exc)})
            progress.empty()
            if ts_rows:
                ts_df = pd.DataFrame(ts_rows).sort_values("Data").reset_index(drop=True)
                try:
                    total_history = save_ndvi_history(ts_rows)
                    st.success(f"Histórico local atualizado ({total_history} registro(s) no arquivo).")
                except Exception as history_exc:
                    st.warning(f"A série foi calculada, mas não foi possível salvar o histórico local: {history_exc}")
                st.session_state["agro_ndvi_timeseries"] = ts_df
                st.session_state["agro_ndvi_timeseries_errors"] = ts_errors
                st.success(f"Série temporal calculada: {len(ts_df)} data(s) com NDVI válido de {len(scenes)} cena(s) consultadas.")
            else:
                st.session_state["agro_ndvi_timeseries"] = pd.DataFrame()
                st.session_state["agro_ndvi_timeseries_errors"] = ts_errors
                st.error("Não foi possível calcular NDVI válido para nenhuma cena no período.")

        ts_df = st.session_state.get("agro_ndvi_timeseries")
        if isinstance(ts_df, pd.DataFrame) and not ts_df.empty:
            chart_df = ts_df.copy()
            chart_df["Data"] = pd.to_datetime(chart_df["Data"], errors="coerce")
            chart_df = chart_df.dropna(subset=["Data"]).set_index("Data")
            daily_chart = chart_df.groupby(chart_df.index.date)[["NDVI médio", "NDVI mediano"]].mean()
            daily_chart.index = pd.to_datetime(daily_chart.index)
            st.line_chart(daily_chart, y_label="NDVI", x_label="Data")
            daily_stats = ts_df.groupby("Data", as_index=False)[["NDVI médio", "NDVI mediano"]].mean().sort_values("Data")
            latest = daily_stats.iloc[-1]
            first = daily_stats.iloc[0]
            delta_mean = float(latest["NDVI médio"] - first["NDVI médio"])
            a, b, c = st.columns(3)
            a.metric("Datas válidas", len(daily_stats))
            b.metric("NDVI médio mais recente", f"{float(latest['NDVI médio']):.3f}", f"{delta_mean:+.3f} desde a primeira data")
            c.metric("NDVI médio na primeira data", f"{float(first['NDVI médio']):.3f}")
            st.dataframe(ts_df, use_container_width=True, hide_index=True)
            st.download_button(
                "⬇️ Baixar série temporal CSV",
                data=ts_df.to_csv(index=False).encode("utf-8-sig"),
                file_name="AgroNDVI_serie_temporal.csv",
                mime="text/csv",
                key="timeseries_csv_download",
            )
            st.caption("Interprete a evolução considerando estádio, cultivar, manejo, condições ambientais e qualidade das cenas. A quantidade de pixels válidos pode variar entre datas após a máscara SCL.")
            ts_errors = st.session_state.get("agro_ndvi_timeseries_errors", [])
            if ts_errors:
                with st.expander(f"Ver {len(ts_errors)} cena(s) que falharam no processamento"):
                    st.dataframe(pd.DataFrame(ts_errors), use_container_width=True, hide_index=True)

        # Persistent history saved on disk, available after closing/reopening the app.
        persistent_history = load_ndvi_history()
        if not persistent_history.empty:
            st.subheader("🗂️ Histórico salvo no computador")
            if "Talhão" in persistent_history.columns:
                field_history = persistent_history[persistent_history["Talhão"].astype(str) == str(selected_field_label)].copy()
            else:
                field_history = persistent_history.copy()
            if not field_history.empty:
                if "Data" in field_history.columns:
                    field_history = field_history.sort_values("Data")
                st.dataframe(field_history, use_container_width=True, hide_index=True)
                st.download_button(
                    "⬇️ Exportar histórico deste talhão (CSV)",
                    data=field_history.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"historico_NDVI_{field_name.replace(' ', '_')}.csv",
                    mime="text/csv",
                    key="persistent_history_download",
                )
            else:
                st.caption("Ainda não há histórico salvo para este talhão.")

elif app_page == "🌿 Processar NDVI":
    st.subheader("Processamento manual de bandas")
    with st.sidebar:
        red_file = st.file_uploader("Banda vermelha — B04 (.tif)", type=["tif", "tiff"], key="red")
        nir_file = st.file_uploader("Infravermelho próximo — B08 (.tif)", type=["tif", "tiff"], key="nir")
        boundary_file = st.file_uploader("Limite do talhão (GeoJSON, opcional)", type=["geojson", "json"], key="boundary")
        image_date = st.date_input("Data da imagem", value=date.today(), key="manual_date")
    if red_file and nir_file:
        try:
            red, red_profile, red_nodata = read_uploaded_raster(red_file)
            nir, nir_profile, nir_nodata = read_uploaded_raster(nir_file)
            nir = align_to_reference(nir, nir_profile, red_profile)
            if red_profile.get("crs") is None or red_profile.get("transform") is None:
                st.error("As imagens precisam conter georreferenciamento (CRS e transformação espacial).")
                st.stop()
            valid = np.isfinite(red) & np.isfinite(nir)
            if red_nodata is not None:
                valid &= red != red_nodata
            if nir_nodata is not None:
                valid &= nir != nir_nodata
            valid &= (red + nir) != 0
            ndvi = np.full(red.shape, np.nan, dtype="float32")
            ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid])
            valid &= np.isfinite(ndvi) & (ndvi >= -1) & (ndvi <= 1)
            ndvi[~valid] = np.nan
            if boundary_file:
                try:
                    boundary = load_geojson_bytes(boundary_file.getvalue()).to_crs(red_profile["crs"])
                    with MemoryFile() as mem:
                        profile = red_profile.copy()
                        profile.update(dtype="float32", count=1, nodata=-9999.0)
                        with mem.open(**profile) as ds:
                            ds.write(np.where(np.isfinite(ndvi), ndvi, -9999).astype("float32"), 1)
                        with mem.open() as ds:
                            clipped, _ = mask(ds, [g.__geo_interface__ for g in boundary.geometry], crop=False, nodata=-9999, filled=True)
                    ndvi = clipped[0].astype("float32")
                    ndvi[ndvi == -9999] = np.nan
                except Exception as exc:
                    st.warning(f"Não foi possível aplicar o limite do talhão: {exc}")
            profile = red_profile.copy()
            profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate")
            show_results(ndvi, profile, field_name, image_date.isoformat(), "Fonte: bandas GeoTIFF enviadas pelo usuário.")
        except Exception as exc:
            st.error(f"Falha no processamento: {exc}")
    else:
        st.info("Envie as bandas B04 e B08 em GeoTIFF para calcular o NDVI manualmente.")

st.divider()
st.caption("AgroNDVI 0.7.0 · NDVI é um indicador de vigor/atividade vegetativa e não diagnostica isoladamente doenças, nematoides ou causas de estresse. Para análises experimentais, registre sensor, data, estádio, cultivar, manejo e critérios de máscara.")
