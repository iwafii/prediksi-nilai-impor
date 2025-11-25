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

# KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Analisis & Prediksi Impor Indonesia",
    layout="wide"
)

# SEMUA FUNGSI LOGIKA (DIAMBIL DARI COLAB)

# FUNGSI 1: MEMUAT DAN MEMBERSIHKAN DATA
def load_and_clean_data(uploaded_file):
    """
    Mengambil file Excel/CSV yang diunggah dan melakukan semua langkah 
    pembersihan dan pra-pemrosesan dari notebook.
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

    # Validasi kolom
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
    df0 = df0.dropna(subset=["year","hs","port"]).copy()
    df0["year"] = pd.to_numeric(df0["year"], errors="coerce").astype("Int64")
    df0["hs"] = df0["hs"].astype(str).str.strip()
    df0["port"] = df0["port"].astype(str).str.strip()
    df0["value_usd"] = pd.to_numeric(df0["value_usd"], errors="coerce").fillna(0.0)
    df0["value_usd"] = df0["value_usd"].clip(lower=0)
    
    # Standarisasi label port (opsional)
    df0["port"] = (df0["port"]
           .str.replace(r"\s+", " ", regex=True)
           .str.strip()
           .str.upper())

    # Pastikan HS dua digit
    df0["hs"] = df0["hs"].astype(str).str.replace(r"\D.*", "", regex=True).str.slice(0, 2)
    df0 = df0[df0["hs"] != ''].copy()

    # Gabungkan duplikat persis
    df0 = (df0.groupby(["year","hs","port"], as_index=False)["value_usd"].sum())
    
    # Buang baris "TOTAL"
    mask_tot = df0["port"].str.contains(r"TOTAL|JUMLAH", case=False, na=False)
    df0 = df0.loc[~mask_tot].copy()
    
    # Membuat df_full (untuk Clustering)
    years = pd.Index(range(int(df0["year"].min()), int(df0["year"].max())+1), name="year")
    hs_all = df0["hs"].dropna().unique()
    ports_all = df0["port"].dropna().unique()

    grid = pd.MultiIndex.from_product([years, hs_all, ports_all], names=["year","hs","port"]).to_frame(index=False)
    df_full = (grid.merge(df0, on=["year","hs","port"], how="left")
                   .assign(value_usd=lambda d: d["value_usd"].fillna(0.0)))
    
    # Membuat df_model (untuk Forecasting)
    active_mask = df_full.groupby(["hs","port"])["value_usd"].transform("sum") > 0
    count_active = df_full.assign(active=(df_full["value_usd"]>0).astype(int)) \
                        .groupby(["hs","port"])["active"].transform("sum")
    quality_mask = count_active >= 3
    
    df_model = df_full[active_mask & quality_mask] \
                     .reset_index(drop=True) \
                     .sort_values(["hs","port","year"])
    
    return df_full, df_model

# FUNGSI 2: MENJALANKAN CLUSTERING
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
    cagr  = (last.values / np.where(first.values==0, 1e-9, first.values)) ** np.where(span==0, 1, 1/np.maximum(span,1)) - 1
    base["cagr"] = cagr
    act = g.assign(active=(g["value_usd"]>0).astype(int)).groupby("port")["active"].mean().reset_index()
    base = base.merge(act, on="port", how="left").rename(columns={"active":"active_ratio"})
    top_hs = df.groupby("hs")["value_usd"].sum().nlargest(5).index
    hs_share = df.groupby(["port","hs"])["value_usd"].sum()
    hs_pivot = (hs_share / hs_share.groupby(level=0).transform("sum")).unstack().fillna(0)
    hs_pivot = hs_pivot.reindex(columns=top_hs).fillna(0) # Reindex to ensure consistent columns
    hs_pivot.columns = [f"share_hs_{h}" for h in hs_pivot.columns]
    out = pd.merge(base, hs_pivot, left_on="port", right_index=True, how="left").fillna(0)
    return out

def describe_row(r):
    """Helper function to label clusters."""
    size = ("sangat besar" if r["median_mean"]>=18
            else "besar"    if r["median_mean"]>=16
            else "menengah" if r["median_mean"]>=12
            else "kecil")
    vol = ("stabil" if r["median_vol"]<12 else "berfluktuasi")
    act = ("sangat aktif" if r["median_active"]>=0.9
           else "cukup aktif" if r["median_active"]>=0.6
           else "jarang aktif")
    grow = ("tumbuh cepat" if r["median_cagr"]>=0.10
            else "tumbuh moderat" if r["median_cagr"]>0
            else "menurun/stagnan")
    if size in ["besar","sangat besar"] and act=="sangat aktif" and vol=="stabil":
        label = "Pelabuhan utama nasional"
    elif grow=="tumbuh cepat" and act!="jarang aktif":
        label = "Pelabuhan berkembang/pertumbuhan tinggi"
    elif act=="jarang aktif":
        label = "Pelabuhan pasif/spesialis"
    else:
        label = "Pelabuhan menengah/regional"
    return f"{label} — skala {size}, {vol}, {act}, {grow}"

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

    # Find best K
    Ks = list(range(2,7))
    sils = []
    for k in Ks:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(Xc)
        sils.append(silhouette_score(Xc, labels))
    best_k = Ks[int(np.argmax(sils))]
    
    # Fit final model
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    ports_feat["cluster"] = kmeans.fit_predict(Xc)

    # Buat Output
    # 1. Cluster Summary
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
    
    # 2. Top 5 HS
    port2cluster = ports_feat[["port","cluster"]]
    tmp = df_full.merge(port2cluster, on="port", how="left")
    hs_sum_by_cluster = tmp.groupby(["cluster","hs"])["value_usd"].sum().reset_index()
    cluster_total_sum = tmp.groupby("cluster")["value_usd"].sum().reset_index(name="cluster_total")
    hs_by_cluster = hs_sum_by_cluster.merge(cluster_total_sum, on="cluster", how="left")
    hs_by_cluster["share"] = hs_by_cluster["value_usd"] / hs_by_cluster["cluster_total"]
    top5_hs = hs_by_cluster.sort_values(["cluster","share"], ascending=[True, False]).groupby("cluster").head(5)
    
    # 3. PCA
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xc)
    df_pca = pd.DataFrame({"PC1": Xp[:,0], "PC2": Xp[:,1],
                           "cluster": ports_feat["cluster"].values, 
                           "port": ports_feat["port"].values})

    return cluster_summary, ports_feat, top5_hs, df_pca

# FUNGSI 3: MENJALANKAN FORECASTING
def make_panel_features(df, lags=(1,2,3,4), roll_windows=(3,5)):
    """Helper function to create features for forecasting."""
    d = df.sort_values(["hs","port","year"]).copy()
    blocks = []
    for (h,p), g in d.groupby(["hs","port"], as_index=False):
        g = g.sort_values("year").copy()
        for L in lags:
            g[f"lag{L}"] = g["value_usd"].shift(L)
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

def forecast_panel_yearly(df_hist, start_year, end_year, lags, roll_windows, scaler, model, X_cols_num, all_X_cols):
    """Helper function to generate iterative future forecasts."""
    hist = df_hist.copy()
    forecasts = []
    for y in range(start_year, end_year+1):
        # Buat fitur dari histori T-1
        ddh, Xh, yh, metah = make_panel_features(hist, lags=lags, roll_windows=roll_windows)
        latest = int(metah["year"].max())
        
        # Ambil baris terakhir (T-1) untuk memprediksi T
        rows_for_next = (metah["year"]==latest)
        Xn = Xh.loc[rows_for_next].copy()
        
        # Scaling
        Xn_scaled = Xn.copy()
        Xn_scaled[X_cols_num] = scaler.transform(Xn_scaled[X_cols_num])
        
        # Pastikan kolom konsisten dengan data training
        Xn_scaled = Xn_scaled.reindex(columns=all_X_cols).fillna(0)

        # Prediksi
        yhat = model.predict(Xn_scaled)
        
        # Susun hasil
        meta_next = metah.loc[rows_for_next, ["hs","port"]].copy()
        pred_df = meta_next.assign(year=y, value_usd=yhat)
        
        # Tambahkan ke daftar & update histori
        forecasts.append(pred_df)
        hist = pd.concat([hist, pred_df], ignore_index=True)
        
    return pd.concat(forecasts, ignore_index=True)

def eval_metrics(y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    return mae, rmse, r2

def run_forecasting(df_model):
    """
    Melatih model XGBoost dan menghasilkan prediksi 5 tahun ke depan.
    """
    LAGS = (1,2,3,4)
    ROLL = (3,5)
    
    dd, X, y, meta = make_panel_features(df_model, lags=LAGS, roll_windows=ROLL)
    
    num_prefix = ("lag","rollmean_","rollstd_","diff","growth")
    X_cols_num = [c for c in X.columns if any([c.startswith(p) for p in num_prefix])]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[X_cols_num] = scaler.fit_transform(X_scaled[X_cols_num])
    
    # Train-test split
    last_year = int(df_model["year"].max())
    n_test_years = 2
    test_years = set(range(last_year - n_test_years + 1, last_year + 1))
    is_test = meta["year"].isin(test_years)

    X_train, X_test = X_scaled[~is_test], X_scaled[is_test]
    y_train, y_test = y[~is_test], y[is_test]

    # Model Training
    xgb = XGBRegressor(
        n_estimators=900, learning_rate=0.045, max_depth=3,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=2.0,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    
    # Evaluasi
    pred_xgb = xgb.predict(X_test)
    mae_x, rmse_x, r2_x = eval_metrics(y_test, pred_xgb)
    eval_overall = pd.DataFrame({
        "model": ["XGBoost"], "MAE": [mae_x], "RMSE": [rmse_x], "R2": [r2_x]
    })
    
    # Buat Forecast
    start_y = last_year + 1
    end_y   = start_y + 4  # 5 tahun ke depan
    
    fcst = forecast_panel_yearly(
        df_hist=df_model, start_year=start_y, end_year=end_y,
        lags=LAGS, roll_windows=ROLL,
        scaler=scaler, model=xgb, X_cols_num=X_cols_num, all_X_cols=X_train.columns
    )
    
    fcst["value_usd"] = np.where(fcst["value_usd"] < 0, 0, fcst["value_usd"])
    
    # Buat data plot nasional
    hist_total = df_model.groupby("year")["value_usd"].sum().reset_index().rename(columns={"value_usd":"actual"})
    f_total    = fcst.groupby("year")["value_usd"].sum().reset_index().rename(columns={"value_usd":"forecast"})
    forecast_nasional = pd.merge(hist_total, f_total, how="outer", on="year").sort_values("year")

    # Konversi tahun ke string untuk plotting
    df_model['year'] = df_model['year'].astype(str)
    fcst['year'] = fcst['year'].astype(str)
    forecast_nasional['year'] = forecast_nasional['year'].astype(str)
    
    return forecast_nasional, fcst, eval_overall

# FUNGSI UTAMA (MAIN) UNTUK MENJALANKAN SEMUA ANALISIS
@st.cache_data
def run_full_analysis(_uploaded_file):
    """
    Fungsi wrapper untuk menjalankan semua langkah analisis.
    Diberi cache agar tidak berjalan ulang jika file tidak berubah.
    """
    
    # 1. Load & Clean
    with st.spinner("Langkah 1/3: Membersihkan dan memproses data... (Ini mungkin perlu waktu)"):
        df_full, df_model = load_and_clean_data(_uploaded_file)
        if df_full is None:
            return None

    # 2. Clustering
    with st.spinner("Langkah 2/3: Melatih model clustering K-Means..."):
        cluster_summary, ports_feat, top5_hs, df_pca = run_clustering(df_full)

    # 3. Forecasting
    with st.spinner("Langkah 3/3: Melatih model forecasting XGBoost..."):
        forecast_nasional, forecast_panel, eval_overall = run_forecasting(df_model)

    st.success("Analisis Selesai!")
    
    # Kembalikan semua hasil dalam satu dictionary
    return {
        "df_model": df_model,
        "cluster_summary": cluster_summary,
        "ports_feat": ports_feat,
        "top5_hs": top5_hs,
        "df_pca": df_pca,
        "forecast_panel": forecast_panel,
        "forecast_nasional": forecast_nasional,
        "eval_overall": eval_overall
    }

# UI (USER INTERFACE) APLIKASI

# Sidebar untuk Upload
st.sidebar.title("🚢 Panel Kontrol")
uploaded_file = st.sidebar.file_uploader(
    "Unggah Dataset Anda (.xlsx atau .csv)",
    type=["xlsx", "csv"]
)

# Tampilan Utama
st.title("🚢 Prediksi Pendapatan Impor Indonesia Berdasarkan Aktivitas Pendapatan Pelabuhan")

# Jika file belum diunggah, tampilkan pesan
if uploaded_file is None:
    st.info("Silakan unggah file dataset mentah Anda di sidebar untuk memulai analisis.")
    st.stop()

# Jika file sudah diunggah, jalankan analisis
# 'data_dict' akan berisi semua hasil (df, tabel, plot)
data_dict = run_full_analysis(uploaded_file)

# Jika analisis gagal (misal file korup), hentikan
if data_dict is None:
    st.error("Analisis gagal. Periksa kembali format file Anda.")
    st.stop()

# MEMBUAT TAB UNTUK NAVIGASI (SAMA SEPERTI KODE SEBELUMNYA)

tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Ringkasan Proyek",
    "📊 Eksplorasi Data (EDA)",
    "🧩 Analisis Clustering",
    "📈 Analisis Forecasting"
])

# TAB 1: RINGKASAN PROYEK
with tab1:
    st.header("Tujuan Penelitian")
    st.markdown("""
    **1. Mengidentifikasi Pola dan Segmentasi Pelabuhan Impor Utama di Indonesia**
    Melalui penerapan algoritma clustering (K-Means), penelitian ini bertujuan untuk mengelompokkan pelabuhan berdasarkan kesamaan pola nilai impor tahunan, volume transaksi, dan fluktuasi antar tahun. Hasilnya memberikan gambaran karakteristik pelabuhan, seperti:
    * Klaster pelabuhan besar dengan nilai impor tinggi dan variasi fluktuatif (misalnya Tanjung Priok dan Belawan),
    * Klaster pelabuhan menengah dengan aktivitas stabil,
    * Klaster pelabuhan kecil dengan nilai impor rendah dan pertumbuhan stagnan.

    **2. Memprediksi Tren Nilai Impor Nasional Periode 2026–2030**
    Dengan memanfaatkan model Extreme Gradient Boosting (XGBoost) berbasis data historis 2014–2025, penelitian ini bertujuan untuk meramalkan tren lima tahun ke depan pada tingkat nasional. Hasil peramalan menunjukkan adanya kecenderungan penurunan moderat nilai impor Indonesia, yang mengindikasikan perbaikan efisiensi rantai pasok dan peningkatan kemandirian produksi dalam negeri.
    """)
    
    st.header("Dataset")
    st.markdown(f"""
    * **Nama File:** `{uploaded_file.name}`
    * **Ukuran File:** `{uploaded_file.size / (1024*1024):.2f} MB`
    * **Data Digunakan:** 17 kode HS yang relevan dengan SDGs 8 (Decent Work and Economic Growth).
    """)
    st.info("Gunakan tab di atas untuk menavigasi hasil analisis.")

    st.header("Detail Kode HS yang Dianalisis")
    st.markdown("""
    Penelitian ini berfokus pada 17 kode HS (Harmonized System) yang dipilih karena relevansinya terhadap **SDGs 8 (Pekerjaan Layak dan Pertumbuhan Ekonomi)**. Kode-kode ini mewakili komponen kunci untuk ekonomi, ketenagakerjaan, industrialisasi, dan produktivitas.
    
    Kode HS yang dianalisis antara lain:
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

# TAB 2: EKSPLORASI DATA (EDA)
with tab2:
    st.header("Eksplorasi Data Historis (Hasil Pembersihan)")

    try:
        # Ambil data model
        df_model_eda = data_dict["df_model"].copy()
        
        # Plot 1: Tren Nasional (Selalu Tampil)
        # Salinan untuk agregasi, pastikan tahun adalah numerik
        df_agg = df_model_eda.copy()
        df_agg['year'] = pd.to_numeric(df_agg['year'])
        annual_total = df_agg.groupby("year")["value_usd"].sum().reset_index()
        annual_total['year'] = annual_total['year'].astype(str) # Balikkan ke string untuk plot

        fig_annual = px.line(
            annual_total, x="year", y="value_usd",
            title="Total Impor Indonesia (USD) per Tahun",
            markers=True, labels={"value_usd": "Total Nilai Impor (USD)", "year": "Tahun"}
        )
        st.plotly_chart(fig_annual, use_container_width=True)
        
        st.divider() # Pemisah visual

        # Filter Tahun
        # Dapatkan daftar tahun unik (sebagai angka) untuk filter
        numeric_years = sorted(pd.to_numeric(df_model_eda['year']).unique())
        year_options = ['Semua Tahun'] + numeric_years
        
        selected_year = st.selectbox(
            "Pilih Tahun Spesifik untuk melihat detail:", 
            year_options
        )

        # Tampilan Dinamis Berdasarkan Filter
        if selected_year == 'Semua Tahun':
            # Tampilkan statistik global
            st.markdown(f"""
            Analisis data dari **{numeric_years[0]}** hingga **{numeric_years[-1]}**:
            * **Total Baris Data Aktif (sepanjang waktu):** `{len(df_model_eda):,}`
            * **Jumlah Kode HS Unik (sepanjang waktu):** `{df_model_eda['hs'].nunique()}`
            * **Jumlah Pelabuhan Unik (sepanjang waktu):** `{df_model_eda['port'].nunique()}`
            """)

            st.subheader("Contoh Data yang Digunakan (Panel Aktif - Acak)")
            st.dataframe(df_model_eda.sample(10, random_state=42), use_container_width=True)
        
        else:
            # Filter data berdasarkan tahun yang dipilih
            # Ingat bahwa kolom 'year' di df_model_eda adalah string
            df_filtered = df_model_eda[df_model_eda['year'] == str(selected_year)]
            
            # Tampilkan statistik untuk tahun tersebut
            st.markdown(f"Analisis data untuk tahun **{selected_year}**:")
            st.markdown(f"""
            * **Total Baris Data Aktif:** `{len(df_filtered):,}`
            * **Jumlah Kode HS Unik:** `{df_filtered['hs'].nunique()}`
            * **Jumlah Pelabuhan Unik:** `{df_filtered['port'].nunique()}`
            * **Total Nilai Impor Tahun Ini:** `USD {df_filtered['value_usd'].sum():,.2f}`
            """)

            st.subheader(f"Contoh Data Panel Aktif (Top 10 Impor - {selected_year})")
            st.dataframe(df_filtered.sort_values('value_usd', ascending=False).head(10), use_container_width=True)

            # Tampilkan Top 10 untuk tahun tersebut
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
                top_hs_year = df_filtered.groupby('hs')['value_usd'].sum().nlargest(10).reset_index()
                fig_hs = px.bar(
                    top_hs_year.sort_values('value_usd', ascending=True), 
                    x='value_usd', y='hs', orientation='h', 
                    title=f'Top 10 Kode HS Impor - {selected_year}',
                    labels={"value_usd": "Total Nilai Impor (USD)", "hs": "Kode HS"}
                )
                st.plotly_chart(fig_hs, use_container_width=True)

    except Exception as e:
        st.error(f"Error saat membuat plot EDA: {e}")
        st.exception(e) # Menambahkan ini untuk debugging yang lebih baik

# TAB 3: ANALISIS CLUSTERING
with tab3:
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
            dominant_hs = data_dict["top5_hs"][data_dict["top5_hs"]['cluster'] == selected_cluster]
            st.dataframe(dominant_hs[['hs', 'share']], use_container_width=True)

            st.markdown(f"**Contoh Pelabuhan di Cluster {selected_cluster}:**")
            ports_in_cluster = data_dict["ports_feat"][
                data_dict["ports_feat"]['cluster'] == selected_cluster
            ][['port', 'port_sum']].sort_values('port_sum', ascending=False).head(10)
            st.dataframe(ports_in_cluster['port'], use_container_width=True)


# TAB 4: ANALISIS FORECASTING
with tab4:
    st.header("Hasil Prediksi (Forecasting) dengan XGBoost")

    st.subheader("Prediksi Tren Nilai Impor Nasional (5 Tahun ke Depan)")
    
    df_nasional = data_dict["forecast_nasional"].copy()
    
    fig_nasional = go.Figure()
    fig_nasional.add_trace(go.Scatter(
        x=df_nasional['year'], y=df_nasional['actual'],
        mode='lines+markers', name='Aktual', line=dict(color='blue')
    ))
    fig_nasional.add_trace(go.Scatter(
        x=df_nasional['year'], y=df_nasional['forecast'],
        mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')
    ))
    fig_nasional.update_layout(
        title="Total Impor Nasional – Aktual vs Forecast (Dilatih Saat Ini)",
        xaxis_title="Tahun", yaxis_title="Total Nilai Impor (USD)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_nasional, use_container_width=True)

    st.subheader("Evaluasi Kinerja Model XGBoost (Dilatih Saat Ini)")
    eval_df = data_dict["eval_overall"]
    xgb_eval = eval_df[eval_df['model'] == 'XGBoost'].iloc[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("R² Score", f"{xgb_eval['R2']:.4f}")
    col2.metric("Mean Absolute Error (MAE)", f"USD {xgb_eval['MAE']/1e6:.2f} Juta")
    col3.metric("Root Mean Squared Error (RMSE)", f"USD {xgb_eval['RMSE']/1e6:.2f} Juta")
    
    st.divider()

    st.subheader("Eksplorasi Prediksi per Kode HS & Pelabuhan")
    
    df_hist_filter = data_dict["df_model"].copy()
    df_fcst_filter = data_dict["forecast_panel"].copy()
    
    df_hist_filter = df_hist_filter[['year', 'hs', 'port', 'value_usd']].rename(columns={"value_usd": "actual"})
    df_fcst_filter = df_fcst_filter[['year', 'hs', 'port', 'value_usd']].rename(columns={"value_usd": "forecast"})
    
    df_combined_filter = pd.merge(df_hist_filter, df_fcst_filter, on=['year', 'hs', 'port'], how='outer')

    col1, col2 = st.columns(2)
    with col1:
        hs_list = ['Semua'] + sorted(df_hist_filter['hs'].unique().tolist())
        selected_hs = st.selectbox("Pilih Kode HS:", hs_list)
    
    with col2:
        if selected_hs == 'Semua':
            ports_list_filtered = ['Semua'] + sorted(df_hist_filter['port'].unique().tolist())
        else:
            ports_list_filtered = ['Semua'] + sorted(df_hist_filter[df_hist_filter['hs'] == selected_hs]['port'].unique().tolist())
        selected_port = st.selectbox("Pilih Pelabuhan:", ports_list_filtered)

    if selected_hs == 'Semua' and selected_port == 'Semua':
        st.info("Anda sedang melihat data nasional. Gunakan plot di atas.")
    else:
        if selected_hs != 'Semua':
            df_combined_filter = df_combined_filter[df_combined_filter['hs'] == selected_hs]
        if selected_port != 'Semua':
            df_combined_filter = df_combined_filter[df_combined_filter['port'] == selected_port]
        
        df_plot_detail = df_combined_filter.groupby('year')[['actual', 'forecast']].sum().reset_index()
        
        fig_detail = go.Figure()
        fig_detail.add_trace(go.Scatter(
            x=df_plot_detail['year'], y=df_plot_detail['actual'],
            mode='lines+markers', name='Aktual', line=dict(color='blue')
        ))
        fig_detail.add_trace(go.Scatter(
            x=df_plot_detail['year'], y=df_plot_detail['forecast'],
            mode='lines+markers', name='Forecast', line=dict(color='red', dash='dash')
        ))
        
        title_detail = f"Aktual vs Forecast untuk: {selected_hs if selected_hs != 'Semua' else 'Semua HS'} | {selected_port if selected_port != 'Semua' else 'Semua Pelabuhan'}"
        fig_detail.update_layout(
            title=title_detail,
            xaxis_title="Tahun", yaxis_title="Nilai Impor (USD)",
            hovermode="x unified"
        )
        st.plotly_chart(fig_detail, use_container_width=True)