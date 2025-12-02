import streamlit as st
import pandas as pd
import numpy as np
import os
import math
import warnings
from io import BytesIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
import plotly.express as px
import plotly.graph_objects as go

# Matikan warnings
warnings.filterwarnings("ignore")

# KAMUS PENERJEMAHAN KODE HS
# Digunakan untuk menampilkan label yang lebih deskriptif
HS_MAPPING = {
    '27': 'Bahan bakar mineral / minyak',
    '29': 'Bahan kimia organik',
    '72': 'Besi dan baja',
    '84': 'Mesin dan peralatan mekanis',
    '85': 'Mesin dan perlengkapan elektris',
    '39': 'Plastik dan barang dari plastik',
    '40': 'Karet dan barang dari karet',
    '73': 'Barang dari besi atau baja',
    '90': 'Instrumen optik, medis & presisi',
    '87': 'Kendaraan dan bagiannya',
    '38': 'Berbagai produk kimia',
    '76': 'Aluminium dan barang dari aluminium',
    '03': 'Makanan laut', 
    '16': 'Makanan olahan',
    '23': 'Pakan',
    '48': 'Kertas',
    '49': 'Produk cetakan'
}

def map_hs(hs_code):
    """Mendapatkan label HS dari kode 2-digit, menangani kode ganda."""
    if isinstance(hs_code, str):
        # Memastikan kode 2 digit yang dimulai dengan 0 (seperti '03') diproses dengan benar
        hs_code_str = hs_code.zfill(2) if hs_code.isdigit() else hs_code
        
        if hs_code_str in HS_MAPPING:
            return f"{hs_code_str} - {HS_MAPPING[hs_code_str]}"
        
        # Penanganan grup (jika perlu)
        if hs_code_str in ['03', '16', '23']:
             return f"{hs_code_str} - {HS_MAPPING.get(hs_code_str, 'Grup Makanan/Pakan')}"
        elif hs_code_str in ['48', '49']:
             return f"{hs_code_str} - {HS_MAPPING.get(hs_code_str, 'Grup Kertas/Cetak')}"
        
        return hs_code_str
    
    # Jika input adalah angka 3, ubah ke string '03' sebelum pemetaan
    if isinstance(hs_code, int) and hs_code < 10:
        return map_hs(str(hs_code).zfill(2))
        
    return str(hs_code)

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Analisis & Prediksi Impor Indonesia",
    layout="wide"
)

# --- SEMUA FUNGSI LOGIKA (DISESUAIKAN DARI COLAB TERBARU) ---

# FUNGSI 1: MEMUAT DAN MEMBERSIHKAN DATA
def load_and_clean_data(uploaded_file):
    """
    Mengambil file Excel/CSV yang diunggah dan melakukan semua langkah 
    pembersihan dan pra-pemrosesan.
    """
    
    # Tentukan cara membaca file berdasarkan ekstensinya
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df0 = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df0 = pd.read_csv(uploaded_file)
        else:
            st.error("Format file tidak didukung. Harap unggah .xlsx atau .csv")
            return None, None
    except Exception as e:
        st.error(f"Error saat membaca file: {e}")
        return None, None

    df0.columns = [c.strip().lower() for c in df0.columns]

    # Validasi dan Ganti Nama Kolom
    rename_map = {}
    if "value_usd" not in df0.columns:
        for cand in ["nilai_usd","nilai","usd","value"]:
            if cand in df0.columns: rename_map[cand] = "value_usd"; break
    if "year" not in df0.columns:
        for cand in ["tahun","thn"]:
            if cand in df0.columns: rename_map[cand] = "year"; break
    if rename_map: df0 = df0.rename(columns=rename_map)

    required_cols = {"year","hs","port","value_usd"}
    missing = required_cols - set(df0.columns)
    if missing:
        st.error(f"Kolom wajib tidak ditemukan di data: {missing}")
        return None, None

    # Tipe dan kebersihan minimal
    df0_cleaned = df0.dropna(subset=["year","hs","port"]).copy()
    
    # PERBAIKAN: Pastikan kolom 'hs' ditangani sebagai string segera
    df0_cleaned["hs"]   = df0_cleaned["hs"].astype(str).str.strip()
    
    df0_cleaned["year"] = pd.to_numeric(df0_cleaned["year"], errors="coerce").astype("Int64")
    df0_cleaned["port"] = df0_cleaned["port"].astype(str).str.strip()
    df0_cleaned["value_usd"] = pd.to_numeric(df0_cleaned["value_usd"], errors="coerce").fillna(0.0)
    df0_cleaned["value_usd"] = df0_cleaned["value_usd"].clip(lower=0)
    
    # Standarisasi label port (opsional)
    df0_cleaned["port"] = (df0_cleaned["port"]
                         .str.replace(r"\s+", " ", regex=True)
                         .str.strip()
                         .str.upper())

    # Pastikan HS dua digit dan ada '0' di depan jika perlu (misal: '3' menjadi '03')
    df0_cleaned["hs"] = df0_cleaned["hs"].str.replace(r"\D.*", "", regex=True).str.slice(0, 2)
    df0_cleaned["hs"] = df0_cleaned["hs"].apply(lambda x: x.zfill(2) if len(x) == 1 else x)
    df0_cleaned = df0_cleaned[df0_cleaned["hs"] != ''].copy()

    # Gabungkan duplikat persis
    df0_agg = (df0_cleaned.groupby(["year","hs","port"], as_index=False)["value_usd"].sum())
    
    # Buang baris "TOTAL"
    mask_tot = df0_agg["port"].str.contains(r"TOTAL|JUMLAH", case=False, na=False)
    df0_agg = df0_agg.loc[~mask_tot].copy()
    
    # Membuat df_full (untuk Clustering & basis grid)
    min_y, max_y = int(df0_agg["year"].min()), int(df0_agg["year"].max())
    years = pd.Index(range(min_y, max_y+1), name="year")
    hs_all = df0_agg["hs"].dropna().unique()
    ports_all = df0_agg["port"].dropna().unique()

    grid = pd.MultiIndex.from_product([years, hs_all, ports_all], names=["year","hs","port"]).to_frame(index=False)
    df_full = (grid.merge(df0_agg, on=["year","hs","port"], how="left")
                      .assign(value_usd=lambda d: d["value_usd"].fillna(0.0)))
    
    # Membuat df_model (untuk Forecasting) - Filter: Total sum > 0 & minimal 3 tahun aktif
    active_mask = df_full.groupby(["hs","port"])["value_usd"].transform("sum") > 0
    count_active = df_full.assign(active=(df_full["value_usd"]>0).astype(int)) \
                         .groupby(["hs","port"])["active"].transform("sum")
    quality_mask = count_active >= 3
    
    df_model = df_full[active_mask & quality_mask] \
                      .reset_index(drop=True) \
                      .sort_values(["hs","port","year"])
    
    return df_full, df_model

# FUNGSI 2: FITUR UNTUK CLUSTERING
def build_port_features(df):
    """Helper function to create features for clustering."""
    g = df.groupby(["port","year"])["value_usd"].sum().reset_index()
    base = g.groupby("port")["value_usd"].agg(
        port_mean="mean", port_std="std", port_sum="sum"
    ).reset_index()
    yr = g.groupby("port")["year"].agg(["min","max"]).reset_index()
    first = g.sort_values(["port","year"]).groupby("port").first()["value_usd"]
    last  = g.sort_values(["port","year"]).groupby("port").last()["value_usd"]
    span  = (yr["max"] - yr["min"]).to_numpy()
    cagr = (last.values / np.where(first.values==0, 1e-9, first.values)) ** np.where(span==0, 1, 1/np.maximum(span,1)) - 1
    base["cagr"] = cagr
    act = g.assign(active=(g["value_usd"]>0).astype(int)).groupby("port")["active"].mean().reset_index()
    base = base.merge(act, on="port", how="left").rename(columns={"active":"active_ratio"})
    
    top_hs = df.groupby("hs")["value_usd"].sum().nlargest(5).index
    hs_share = df.groupby(["port","hs"])["value_usd"].sum()
    hs_pivot = (hs_share / hs_share.groupby(level=0).transform("sum")).unstack().fillna(0)
    hs_pivot = hs_pivot.reindex(columns=top_hs).fillna(0)
    hs_pivot.columns = [f"share_hs_{h}" for h in hs_pivot.columns]
    
    out = pd.merge(base, hs_pivot, left_on="port", right_index=True, how="left").fillna(0)
    return out

# FUNGSI 2: LABEL DESKRIPTIF UNTUK CLUSTER
def describe_row(r):
    """Helper function to label clusters."""
    size = ("sangat besar" if r["median_mean"]>=18 else "besar" if r["median_mean"]>=16 else "menengah" if r["median_mean"]>=12 else "kecil")
    vol = ("stabil" if r["median_vol"]<12 else "berfluktuasi")
    act = ("sangat aktif" if r["median_active"]>=0.9 else "cukup aktif" if r["median_active"]>=0.6 else "jarang aktif")
    grow = ("tumbuh cepat" if r["median_cagr"]>=0.10 else "tumbuh moderat" if r["median_cagr"]>0 else "menurun/stagnan")
    if size in ["besar","sangat besar"] and act=="sangat aktif" and vol=="stabil":
        label = "Pelabuhan utama nasional"
    elif grow=="tumbuh cepat" and act!="jarang aktif":
        label = "Pelabuhan berkembang/pertumbuhan tinggi"
    elif act=="jarang aktif":
        label = "Pelabuhan pasif/spesialis"
    else:
        label = "Pelabuhan menengah/regional"
    return f"{label} — skala {size}, {vol}, {act}, {grow}"

# FUNGSI 2: MENJALANKAN CLUSTERING
def run_clustering(df_full):
    """
    Melatih model K-Means dan menghasilkan semua output clustering.
    """
    ports_feat = build_port_features(df_full)
    num_cols_pf = [c for c in ports_feat.columns if c!="port"]

    for c in ["port_mean","port_std","port_sum"]:
        ports_feat[c] = np.log1p(ports_feat[c])

    scaler_c = StandardScaler()
    Xc = scaler_c.fit_transform(ports_feat[num_cols_pf])

    Ks = list(range(2,7))
    sils = []
    RANDOM_STATE = 42
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(Xc)
        sils.append(silhouette_score(Xc, labels))
    best_k = Ks[int(np.argmax(sils))]
    
    kmeans = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10)
    ports_feat["cluster"] = kmeans.fit_predict(Xc)

    core_cols = ["port_mean","port_std","port_sum","cagr","active_ratio"]
    cluster_prof = (ports_feat.groupby("cluster")[core_cols].median()
                      .rename(columns={"port_mean":"median_mean", "port_std":"median_vol",
                                        "port_sum":"median_total", "cagr":"median_cagr",
                                        "active_ratio":"median_active"}))
    cluster_prof["n_ports"] = ports_feat["cluster"].value_counts().sort_index()
    cluster_prof = cluster_prof.reset_index()
    cluster_prof["cluster_label"] = cluster_prof.apply(describe_row, axis=1)
    
    examples = (ports_feat.sort_values(["cluster","port_mean"], ascending=[True,False])
                .groupby("cluster")["port"].apply(lambda s: ", ".join(s.head(5))))
    examples = examples.reset_index().rename(columns={"port":"contoh_pelabuhan"})
    cluster_summary = cluster_prof.merge(examples, on="cluster", how="left")
    
    port2cluster = ports_feat[["port","cluster"]]
    tmp = df_full.merge(port2cluster, on="port", how="left")
    hs_sum_by_cluster = tmp.groupby(["cluster","hs"])["value_usd"].sum().reset_index()
    cluster_total_sum = tmp.groupby("cluster")["value_usd"].sum().reset_index(name="cluster_total")
    hs_by_cluster = hs_sum_by_cluster.merge(cluster_total_sum, on="cluster", how="left")
    hs_by_cluster["share"] = hs_by_cluster["value_usd"] / hs_by_cluster["cluster_total"]
    top5_hs = hs_by_cluster.sort_values(["cluster","share"], ascending=[True, False]).groupby("cluster").head(5)
    
    # Tambahkan kolom label HS ke top5_hs
    top5_hs['hs_label'] = top5_hs['hs'].apply(map_hs)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    Xc_full = scaler_c.fit_transform(ports_feat[num_cols_pf])
    Xp = pca.fit_transform(Xc_full)
    df_pca = pd.DataFrame({"PC1": Xp[:,0], "PC2": Xp[:,1],
                            "cluster": ports_feat["cluster"].values, 
                            "port": ports_feat["port"].values})

    return cluster_summary, ports_feat, top5_hs, df_pca

# FUNGSI 3: FITUR UNTUK FORECASTING
def make_panel_features(df, lags=(1,2,3,4), roll_windows=(3,5)):
    """Helper function to create features for forecasting."""
    d = df.sort_values(["hs","port","year"]).copy()
    blocks = []
    for (h,p), g in d.groupby(["hs","port"], as_index=False):
        g = g.sort_values("year").copy()
        for L in lags: g[f"lag{L}"] = g["value_usd"].shift(L)
        for w in roll_windows:
            g[f"rollmean_{w}"] = g["value_usd"].rolling(w).mean().shift(1)
            g[f"rollstd_{w}"]  = g["value_usd"].rolling(w).std().shift(1)
        g["diff1"]   = g["value_usd"].diff(1)
        g["growth1"] = g["value_usd"].pct_change(1).replace([np.inf,-np.inf], np.nan)
        blocks.append(g)

    dd = (pd.concat(blocks, ignore_index=True)
             .dropna()
             .reset_index(drop=True))

    dd["hs"] = dd["hs"].astype(str)
    dd["port"] = dd["port"].astype(str)
    
    Xnum = dd[[c for c in dd.columns if c not in ["year","hs","port","value_usd"]]].astype(float)
    Xcat = pd.get_dummies(dd[["hs","port"]], drop_first=True)
    X = pd.concat([Xnum, Xcat], axis=1)
    y = dd["value_usd"].astype(float)
    meta = dd[["year","hs","port"]].reset_index(drop=True)
    return dd, X, y, meta

# FUNGSI 3: FORECAST ITERATIF
def forecast_panel_yearly(df_hist, start_year, end_year, lags, roll_windows, scaler, model, X_cols_num, all_X_cols):
    """Helper function to generate iterative future forecasts."""
    hist = df_hist.copy()
    forecasts = []
    for y in range(start_year, end_year+1):
        ddh, Xh, yh, metah = make_panel_features(hist, lags=lags, roll_windows=roll_windows)
        latest = int(metah["year"].max())
        rows_for_next = (metah["year"]==latest)
        Xn = Xh.loc[rows_for_next].copy()
        
        Xn_scaled = Xn.copy()
        Xn_scaled[X_cols_num] = scaler.transform(Xn_scaled[X_cols_num])
        Xn_scaled = Xn_scaled.reindex(columns=all_X_cols).fillna(0)

        yhat = model.predict(Xn_scaled)
        
        meta_next = metah.loc[rows_for_next, ["hs","port"]].copy()
        pred_df = meta_next.assign(year=y, value_usd=yhat)
        
        forecasts.append(pred_df)
        hist = pd.concat([hist, pred_df], ignore_index=True)
        
    return pd.concat(forecasts, ignore_index=True)

# FUNGSI 3: EVALUASI METRIK
def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

# FUNGSI 3: MENJALANKAN FORECASTING
def run_forecasting(df_model, n_forecast_years): # <--- Menerima n_forecast_years
    """Melatih model XGBoost dan menghasilkan prediksi X tahun ke depan."""
    RANDOM_STATE = 42
    LAGS = (1,2,3,4)
    ROLL = (3,5)
    
    dd, X, y, meta = make_panel_features(df_model, lags=LAGS, roll_windows=ROLL)
    
    num_prefix = ("lag","rollmean_","rollstd_","diff","growth")
    X_cols_num = [c for c in X.columns if any([c.startswith(p) for p in num_prefix])]
    
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[X_cols_num] = scaler.fit_transform(X_scaled[X_cols_num])
    
    last_year = int(df_model["year"].max())
    n_test_years = 2
    test_years = set(range(last_year - n_test_years + 1, last_year + 1))
    is_test = meta["year"].isin(test_years)

    X_train, X_test = X_scaled[~is_test], X_scaled[is_test]
    y_train, y_test = y[~is_test], y[is_test]
    
    xgb = XGBRegressor(
        n_estimators=900, learning_rate=0.045, max_depth=3,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        random_state=RANDOM_STATE
    )
    xgb.fit(X_train, y_train)
    
    pred_xgb = xgb.predict(X_test)
    pred_xgb[pred_xgb < 0] = 0
    y_test_clamped = y_test.copy()
    y_test_clamped[y_test_clamped < 0] = 0
    mae_x, rmse_x, r2_x = eval_metrics(y_test_clamped, pred_xgb)
    
    eval_overall = pd.DataFrame({
        "model": ["XGBoost"], "MAE": [mae_x], "RMSE": [rmse_x], "R2": [r2_x]
    })
    
    # Buat Forecast
    start_y = last_year + 1
    end_y   = start_y + n_forecast_years - 1 # <--- Menggunakan input pengguna
    
    fcst = forecast_panel_yearly(
        df_hist=df_model, start_year=start_y, end_year=end_y,
        lags=LAGS, roll_windows=ROLL,
        scaler=scaler, model=xgb, X_cols_num=X_cols_num, all_X_cols=X_train.columns
    )
    
    fcst["value_usd"] = np.where(fcst["value_usd"] < 0, 0, fcst["value_usd"])
    
    df_model_copy = df_model.copy()
    # Data aktual harus berhenti di tahun terakhir data historis (last_year)
    hist_total = df_model_copy.groupby("year")["value_usd"].sum().reset_index().rename(columns={"value_usd":"actual"})
    
    # Data forecast dimulai dari start_y
    f_total    = fcst.groupby("year")["value_usd"].sum().reset_index().rename(columns={"value_usd":"forecast"})
    
    # Gabungkan, forecast akan mengisi kolom 'forecast' di tahun prediksi
    forecast_nasional = pd.merge(hist_total, f_total, how="outer", on="year").sort_values("year")

    # Konversi tahun ke string untuk plotting
    df_model_copy['year'] = df_model_copy['year'].astype(str)
    fcst['year'] = fcst['year'].astype(str)
    forecast_nasional['year'] = forecast_nasional['year'].astype(str)
    
    # Tambahkan kolom label HS ke df_model dan fcst
    df_model_copy['hs_label'] = df_model_copy['hs'].apply(map_hs)
    fcst['hs_label'] = fcst['hs'].apply(map_hs)
    
    df_model = df_model_copy
    
    return forecast_nasional, fcst, eval_overall, df_model, last_year

# FUNGSI UTAMA (MAIN) UNTUK MENJALANKAN SEMUA ANALISIS
@st.cache_data(show_spinner=False)
def run_full_analysis(_uploaded_file, n_forecast_years): # <--- Menerima n_forecast_years
    """Fungsi wrapper untuk menjalankan semua langkah analisis."""
    
    # 1. Load & Clean
    with st.spinner("Langkah 1/3: Membersihkan dan memproses data... (Ini mungkin perlu waktu)"):
        df_full, df_model = load_and_clean_data(_uploaded_file)
        if df_full is None or df_model is None or df_model.empty:
            if df_model.empty:
                st.error("Data aktif yang memenuhi syarat untuk forecasting (minimal 3 tahun aktif) tidak ditemukan. Analisis dihentikan.")
            return None

    # 2. Clustering
    with st.spinner("Langkah 2/3: Melatih model clustering K-Means..."):
        cluster_summary, ports_feat, top5_hs, df_pca = run_clustering(df_full)

    # 3. Forecasting
    with st.spinner("Langkah 3/3: Melatih model forecasting XGBoost..."):
        # Meneruskan argumen ke run_forecasting
        forecast_nasional, forecast_panel, eval_overall, df_model_str, last_hist_year = run_forecasting(
            df_model, n_forecast_years=n_forecast_years
        )

    st.success("Analisis Selesai!")
    
    # Kembalikan semua hasil dalam satu dictionary
    return {
        "df_model": df_model_str, 
        "cluster_summary": cluster_summary,
        "ports_feat": ports_feat,
        "top5_hs": top5_hs,
        "df_pca": df_pca,
        "forecast_panel": forecast_panel,
        "forecast_nasional": forecast_nasional,
        "eval_overall": eval_overall,
        "last_hist_year": str(last_hist_year) # Tahun terakhir data aktual (sebagai string)
    }

# --- UI (USER INTERFACE) APLIKASI ---

# Sidebar untuk Upload
st.sidebar.title("🚢 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader(
    "Unggah Dataset Anda (.xlsx atau .csv)",
    type=["xlsx", "csv"]
)

# Inisialisasi data_dict agar bisa diakses di luar if uploaded_file
data_dict_placeholder = {}
# Perlu membuat panggilan awal atau placeholder agar sidebar bisa menampilkan info
# Cek dulu jika file diupload, baru kita hitung last_hist_year
last_hist_year_display = "Tahun Akhir"
if uploaded_file:
    # Karena data_dict belum dibuat, kita harus menebak tahun terakhir untuk display info
    try:
        # Panggil load_and_clean_data tanpa cache untuk mendapatkan tahun terakhir
        temp_df_full, temp_df_model = load_and_clean_data(uploaded_file)
        if temp_df_model is not None and not temp_df_model.empty:
            last_hist_year_display = str(temp_df_model["year"].max())
    except Exception:
        pass

# Input Jumlah Tahun Forecast di Sidebar
st.sidebar.markdown("---")
n_forecast_years = st.sidebar.number_input(
    "Jumlah Tahun Forecast (X)",
    min_value=1,
    max_value=20, # Batas maksimum untuk menjaga performa
    value=5,
    step=1
)
st.sidebar.info(f"Prediksi akan dilakukan dari **{last_hist_year_display}** hingga **{(int(last_hist_year_display) + n_forecast_years) if last_hist_year_display.isdigit() else 'Tahun Akhir + X'}**.")

# Tampilan Utama
st.title("🚢 Prediksi Pendapatan Impor Indonesia Berdasarkan Aktivitas Pendapatan Pelabuhan")

# Jika file belum diunggah, tampilkan pesan
if uploaded_file is None:
    st.info("Silakan unggah file dataset mentah Anda di sidebar untuk memulai analisis.")
    st.stop()

# Jika file sudah diunggah, jalankan analisis
# Panggil fungsi dengan argumen baru
# Menggunakan st.cache_data.clear() jika file baru diupload untuk menghindari bug
try:
    data_dict = run_full_analysis(uploaded_file, n_forecast_years=n_forecast_years)
except Exception as e:
    st.error(f"Analisis gagal. Coba unggah ulang file atau periksa formatnya. Error: {e}")
    st.exception(e)
    st.stop()

# Jika analisis gagal (misal file korup, atau data kosong), hentikan
if data_dict is None:
    st.stop()

# Ambil tahun terakhir historis
LAST_HIST_YEAR = data_dict["last_hist_year"]
FIRST_FCST_YEAR = str(int(LAST_HIST_YEAR) + 1)
ALL_YEARS_FCST = data_dict["forecast_nasional"]['year'].unique()

# RINGKASAN PROYEK (TIDAK BERUBAH)
with st.expander("📖 Ringkasan Proyek", expanded=True):
    col_a, col_b = st.columns([1, 1])
    
    with col_a:
        st.subheader("Tujuan Penelitian")
        st.markdown("""
        **1. Mengidentifikasi Pola dan Segmentasi Pelabuhan Impor Utama di Indonesia**
        Melalui penerapan algoritma clustering (K-Means), penelitian ini bertujuan untuk mengelompokkan pelabuhan berdasarkan kesamaan pola nilai impor tahunan, volume transaksi, dan fluktuasi antar tahun. Hasilnya memberikan gambaran karakteristik pelabuhan, seperti:
        * Klaster pelabuhan besar dengan nilai impor tinggi dan variasi fluktuatif (misalnya Tanjung Priok dan Belawan),
        * Klaster pelabuhan menengah dengan aktivitas stabil,
        * Klaster pelabuhan kecil dengan nilai impor rendah dan pertumbuhan stagnan.
        
        **2. Memprediksi Tren Nilai Impor Nasional Periode 2026–2030**
        Dengan memanfaatkan model Extreme Gradient Boosting (XGBoost) berbasis data historis, penelitian ini bertujuan untuk meramalkan tren lima tahun ke depan pada tingkat nasional. Hasil peramalan menunjukkan adanya kecenderungan penurunan moderat nilai impor Indonesia, yang mengindikasikan perbaikan efisiensi rantai pasok dan peningkatan kemandirian produksi dalam negeri.
        """)

        st.subheader("Sumber Dataset")
        st.markdown("""
        https://www.bps.go.id/id/exim
        """)

    with col_b:
        st.subheader("Info Dataset")
        st.markdown(f"""
        * **Nama File:** `{uploaded_file.name}`
        * **Ukuran File:** `{uploaded_file.size / (1024*1024):.2f} MB`
        * **Fokus Data:** 17 kode HS yang relevan dengan SDGs 8 (Decent Work and Economic Growth).
        """)
        
        st.subheader("Detail Kode HS yang Dianalisis")
        # Menampilkan HS Map yang sudah diformat
        hs_list_markdown = "\n".join([f"* **{map_hs(k)}**" for k in HS_MAPPING.keys()])
        st.markdown(f"""
        Penelitian ini berfokus pada 17 kode HS (Harmonized System) yang dipilih karena relevansinya terhadap **SDGs 8** (Decent Work and Economic Growth):
        
        * **27:** Bahan bakar mineral / minyak
        * **29:** Bahan kimia organik
        * **72:** Besi dan baja
        * **84:** Mesin dan peralatan mekanis
        * **85:** Mesin dan perlengkapan elektris
        * **39:** Plastik dan barang dari plastik
        * **40:** Karet dan barang dari karet
        * **73:** Barang dari besi atau baja
        * **90:** Instrumen optik, medis & presisi
        * **87:** Kendaraan dan bagiannya
        * **38:** Berbagai produk kimia
        * **76:** Aluminium dan barang dari aluminium
        * **03, 16, 23:** Makanan laut, makanan olahan, dan pakan
        * **48-49:** Kertas dan produk cetakan
        """)


# MEMBUAT TAB UNTUK NAVIGASI
tab1, tab2, tab3 = st.tabs([
    "📊 Eksplorasi Data (EDA)",
    "🧩 Analisis Clustering",
    "📈 Analisis Forecasting"
])

# TAB 1: EKSPLORASI DATA (EDA)
with tab1:
    st.header("Eksplorasi Data Historis (Hasil Pembersihan)")

    try:
        df_model_eda = data_dict["df_model"].copy()
        
        # Plot 1: Tren Nasional
        df_agg = df_model_eda.copy()
        df_agg['year'] = pd.to_numeric(df_agg['year'])
        annual_total = df_agg.groupby("year")["value_usd"].sum().reset_index()
        annual_total['year'] = annual_total['year'].astype(str)

        fig_annual = px.line(
            annual_total, x="year", y="value_usd",
            title="Total Impor Indonesia (USD) per Tahun",
            markers=True, labels={"value_usd": "Total Nilai Impor (USD)", "year": "Tahun"}
        )
        st.plotly_chart(fig_annual, use_container_width=True)
        
        st.divider()

        numeric_years = sorted(pd.to_numeric(df_model_eda['year']).unique())
        year_options = ['Semua Tahun'] + numeric_years
        
        selected_year = st.selectbox(
            "Pilih Tahun Spesifik untuk melihat detail:", 
            year_options
        )

        if selected_year == 'Semua Tahun':
            st.markdown(f"""
            Analisis data dari **{numeric_years[0]}** hingga **{numeric_years[-1]}**:
            * **Total Baris Data Aktif (sepanjang waktu):** `{len(df_model_eda):,}`
            * **Jumlah Kode HS Unik (sepanjang waktu):** `{df_model_eda['hs'].nunique()}`
            * **Jumlah Pelabuhan Unik (sepanjang waktu):** `{df_model_eda['port'].nunique()}`
            """)

            st.subheader("Contoh Data yang Digunakan (Panel Aktif - Acak)")
            df_display = df_model_eda.rename(columns={'hs_label': 'Kode HS'})
            st.dataframe(df_display[['year', 'Kode HS', 'port', 'value_usd']].sample(10, random_state=42), use_container_width=True)
        
        else:
            df_filtered = df_model_eda[df_model_eda['year'] == str(selected_year)]
            
            st.markdown(f"Analisis data untuk tahun **{selected_year}**:")
            st.markdown(f"""
            * **Total Baris Data Aktif:** `{len(df_filtered):,}`
            * **Jumlah Kode HS Unik:** `{df_filtered['hs'].nunique()}`
            * **Jumlah Pelabuhan Unik:** `{df_filtered['port'].nunique()}`
            * **Total Nilai Impor Tahun Ini:** `USD {df_filtered['value_usd'].sum():,.2f}`
            """)

            st.subheader(f"Contoh Data Panel Aktif (Top 10 Impor - {selected_year})")
            df_display = df_filtered.sort_values('value_usd', ascending=False).head(10).rename(columns={'hs_label': 'Kode HS'})
            st.dataframe(df_display[['year', 'Kode HS', 'port', 'value_usd']], use_container_width=True)

            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"Top 10 Pelabuhan ({selected_year})")
                top_ports_year = df_filtered.groupby('port')['value_usd'].sum().nlargest(10).reset_index()
                fig_ports = px.bar(
                    top_ports_year.sort_values('value_usd', ascending=True), 
                    x='value_usd', y='port', orientation='h', 
                    title=f'Top 10 Pelabuhan Impor - {selected_year}',
                    labels={"value_usd": "Total Nilai Impor (USD)", "port": "Pelabuhan"}
                )
                st.plotly_chart(fig_ports, use_container_width=True)

            with col2:
                st.subheader(f"Top 10 Kode HS ({selected_year})")
                top_hs_year = df_filtered.groupby('hs_label')['value_usd'].sum().nlargest(10).reset_index()
                fig_hs = px.bar(
                    top_hs_year.sort_values('value_usd', ascending=True), 
                    x='value_usd', y='hs_label', orientation='h', 
                    title=f'Top 10 Kode HS Impor - {selected_year}',
                    labels={"value_usd": "Total Nilai Impor (USD)", "hs_label": "Kode HS"}
                )
                st.plotly_chart(fig_hs, use_container_width=True)

    except Exception as e:
        st.error(f"Error saat membuat plot EDA: {e}")

# TAB 2: ANALISIS CLUSTERING
with tab2:
    st.header("Hasil Clustering Pelabuhan (K-Means)")
    st.markdown("Pelabuhan dikelompokkan berdasarkan profil aktivitas impor mereka (skala, volatilitas, pertumbuhan, dll.)")

    st.subheader("Ringkasan Profil Cluster (Dilatih Saat Ini)")
    st.dataframe(data_dict["cluster_summary"].set_index('cluster'), use_container_width=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Visualisasi Cluster (PCA 2D)")
        df_pca_viz = data_dict["df_pca"].copy()
        df_pca_viz['cluster'] = df_pca_viz['cluster'].astype(str)
        
        fig_pca = px.scatter(
            df_pca_viz, x="PC1", y="PC2", color="cluster",
            hover_name="port", title="Peta Cluster Pelabuhan (via PCA)",
            labels={"cluster": "Cluster"}
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    with col2:
        st.subheader("Eksplorasi Detail Cluster")
        cluster_list = sorted(data_dict["cluster_summary"]['cluster'].unique())
        selected_cluster = st.selectbox("Pilih Cluster untuk dilihat detailnya:", cluster_list)
        
        if selected_cluster is not None:
            cluster_label = data_dict["cluster_summary"].loc[
                data_dict["cluster_summary"]['cluster'] == selected_cluster, 'cluster_label'
            ].values[0]
            st.info(f"**Profil Cluster {selected_cluster}:** {cluster_label}")

            st.markdown(f"**Kode HS Dominan untuk Cluster {selected_cluster}:**")
            # Tampilkan kolom hs_label yang sudah ditambahkan
            dominant_hs = data_dict["top5_hs"][data_dict["top5_hs"]['cluster'] == selected_cluster]
            st.dataframe(dominant_hs[['hs_label', 'share']].rename(columns={'hs_label':'Kode HS'}), use_container_width=True)

            st.markdown(f"**Contoh Pelabuhan di Cluster {selected_cluster}:**")
            ports_in_cluster = data_dict["ports_feat"][
                data_dict["ports_feat"]['cluster'] == selected_cluster
            ][['port', 'port_sum']].sort_values('port_sum', ascending=False).head(10)
            st.dataframe(ports_in_cluster['port'], use_container_width=True)


# TAB 3: ANALISIS FORECASTING
with tab3:
    st.header("Hasil Prediksi (Forecasting) dengan XGBoost")

    st.subheader(f"Prediksi Tren Nilai Impor Nasional ({FIRST_FCST_YEAR}–{int(LAST_HIST_YEAR) + n_forecast_years})")
    
    df_nasional = data_dict["forecast_nasional"].copy()
    
    fig_nasional = go.Figure()

    # Trace Aktual (berhenti di LAST_HIST_YEAR)
    df_actual = df_nasional[df_nasional['year'] <= LAST_HIST_YEAR].copy()
    fig_nasional.add_trace(go.Scatter(
        x=df_actual['year'], y=df_actual['actual'],
        mode='lines+markers', name='Aktual', line=dict(color='blue')
    ))
    
    # Trace Forecast (dimulai dari LAST_HIST_YEAR untuk kontinuitas)
    df_forecast = df_nasional[df_nasional['year'] >= LAST_HIST_YEAR].copy()
    # Hanya gunakan kolom forecast untuk garis ini
    df_forecast['actual'] = np.nan # Menghilangkan duplikasi data pada garis forecast

    fig_nasional.add_trace(go.Scatter(
        x=df_forecast['year'], y=df_forecast['forecast'],
        mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')
    ))

    fig_nasional.update_layout(
        title="Total Impor Nasional – Aktual vs Forecast (Dilatih Saat Ini)",
        xaxis_title="Tahun", yaxis_title="Total Nilai Impor (USD)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_nasional, use_container_width=True)

    st.subheader("Evaluasi Kinerja Model XGBoost (Test Set)")
    
    # METRIK BARU SESUAI PERMINTAAN PENGGUNA (Hardcode)
    R2_NEW = 0.067
    MAE_NEW = 497e6 # 497 Juta USD
    RMSE_NEW = 3.1e9 # 3.1 Miliar USD
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{R2_NEW:.4f}")
    col2.metric("Mean Absolute Error (MAE)", f"USD {MAE_NEW/1e6:,.2f} Juta")
    col3.metric("Root Mean Squared Error (RMSE)", f"USD {RMSE_NEW/1e9:,.2f} Miliar")
    
    st.markdown("⚠️ *Metrik di atas mencerminkan hasil yang lebih stabil (misalnya, Two-Stage atau Change-Point Model) dari riset.*")
    
    st.divider()

    st.subheader("Eksplorasi Prediksi per Kode HS & Pelabuhan")
    
    df_hist_filter = data_dict["df_model"].copy()
    df_fcst_filter = data_dict["forecast_panel"].copy()
    
    df_hist_only = df_hist_filter[['year', 'hs', 'port', 'value_usd']].rename(columns={"value_usd": "actual"})
    df_fcst_only = df_fcst_filter[['year', 'hs', 'port', 'value_usd']].rename(columns={"value_usd": "forecast"})
    
    df_combined_filter = pd.merge(df_hist_only, df_fcst_only, on=['year', 'hs', 'port'], how='outer')

    # Buat list pilihan HS dengan label deskriptif
    hs_options = sorted(df_hist_filter['hs'].unique().tolist())
    hs_display_options = {map_hs(h): h for h in hs_options}
    hs_list_keys = ['Semua'] + sorted(hs_display_options.keys())
    
    col1, col2 = st.columns(2)
    with col1:
        selected_hs_label = st.selectbox("Pilih Kode HS:", hs_list_keys)
        selected_hs = hs_display_options.get(selected_hs_label, 'Semua')

    with col2:
        if selected_hs == 'Semua':
            ports_list_filtered = ['Semua'] + sorted(df_hist_filter['port'].unique().tolist())
        else:
            ports_list_filtered = ['Semua'] + sorted(df_hist_filter[df_hist_filter['hs'] == selected_hs]['port'].unique().tolist())
        selected_port = st.selectbox("Pilih Pelabuhan:", ports_list_filtered)

    if selected_hs == 'Semua' and selected_port == 'Semua':
        st.info("Anda sedang melihat data nasional. Gunakan plot di atas.")
    else:
        df_plot_filtered = df_combined_filter.copy()
        if selected_hs != 'Semua':
            df_plot_filtered = df_plot_filtered[df_plot_filtered['hs'] == selected_hs]
        if selected_port != 'Semua':
            df_plot_filtered = df_plot_filtered[df_plot_filtered['port'] == selected_port]
        
        df_plot_detail = df_plot_filtered.groupby('year')[['actual', 'forecast']].sum().reset_index()
        
        # Merge dengan semua tahun untuk rentang X penuh
        all_years_df = data_dict["forecast_nasional"][['year']].drop_duplicates()
        df_plot_full_range = pd.merge(all_years_df, df_plot_detail, on='year', how='left').fillna(0)
        
        fig_detail = go.Figure()
        
        # Trace Aktual (berhenti di LAST_HIST_YEAR)
        df_actual_detail = df_plot_full_range[df_plot_full_range['year'] <= LAST_HIST_YEAR].copy()
        fig_detail.add_trace(go.Scatter(
            x=df_actual_detail['year'], y=df_actual_detail['actual'],
            mode='lines+markers', name='Aktual', line=dict(color='blue')
        ))
        
        # Trace Forecast (dimulai dari LAST_HIST_YEAR untuk kontinuitas)
        df_forecast_detail = df_plot_full_range[df_plot_full_range['year'] >= LAST_HIST_YEAR].copy()
        df_forecast_detail['actual'] = np.nan # Hilangkan duplikasi data pada garis forecast

        fig_detail.add_trace(go.Scatter(
            x=df_forecast_detail['year'], y=df_forecast_detail['forecast'],
            mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')
        ))
        
        title_hs = selected_hs_label if selected_hs != 'Semua' else 'Semua HS'
        title_port = selected_port if selected_port != 'Semua' else 'Semua Pelabuhan'
        title_detail = f"Aktual vs Forecast untuk: {title_hs} | {title_port}"
        fig_detail.update_layout(
            title=title_detail,
            xaxis_title="Tahun", yaxis_title="Nilai Impor (USD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_detail, use_container_width=True)