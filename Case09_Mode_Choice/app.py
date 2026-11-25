import streamlit as st
import pandas as pd
import joblib
import os

# =====================================
# KONFIGURASI
# =====================================

st.set_page_config(
    page_title="Mode Choice Prediction",
    page_icon="🚆",
    layout="wide"
)

MODEL_PATH = "models/random_forest_mode_choice.joblib"

# =====================================
# MEMUAT MODEL
# =====================================

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "File model tidak ditemukan. Jalankan latih.py terlebih dahulu."
        )

    return joblib.load(MODEL_PATH)


try:
    artifact = load_model()

    model = artifact["pipeline"]
    metadata = artifact["metadata"]

    features = metadata["feature_names"]
    metrics = metadata["metrics"]
    classes = metadata["class_labels"]

except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()


# =====================================
# HEADER
# =====================================

st.title("🚆 Mode Choice Prediction")

st.write(
    "Sistem klasifikasi pilihan moda transportasi "
    "menggunakan algoritma Random Forest."
)

st.caption("Dataset: kasus09_mode_choice.csv")

st.divider()


# =====================================
# DASHBOARD EVALUASI
# =====================================

st.subheader("Dashboard Evaluasi Model")

st.write(f"Model yang digunakan: **{metadata['model_name']}**")

col1, col2, col3, col4 = st.columns(4)

col1.write("Accuracy")
col1.subheader(f"{metrics['accuracy'] * 100:.2f}%")

col2.write("Precision")
col2.subheader(f"{metrics['precision_macro'] * 100:.2f}%")

col3.write("Recall")
col3.subheader(f"{metrics['recall_macro'] * 100:.2f}%")

col4.write("F1-Score")
col4.subheader(f"{metrics['f1_macro'] * 100:.2f}%")

st.caption(
    "Metrik merupakan hasil evaluasi akhir menggunakan data testing."
)

st.divider()


# =====================================
# FORM INPUT PREDIKSI
# =====================================

st.subheader("Prediksi Pilihan Moda Transportasi")

st.write(
    "Masukkan karakteristik individu dan atribut dari "
    "keempat alternatif transportasi yang tersedia."
)

st.info(
    "Satu pengisian mewakili satu individu dengan empat alternatif "
    "transportasi. Masukkan nilai sesuai data yang ingin diprediksi."
)

input_data = {}

with st.form("form_prediksi"):

    # ---------------------------------
    # A. KARAKTERISTIK INDIVIDU
    # ---------------------------------

    st.markdown("### A. Karakteristik Individu")

    col1, col2 = st.columns(2)

    with col1:
        input_data["hinc"] = st.number_input(
            "Pendapatan Rumah Tangga (hinc)",
            value=35.0,
            step=0.1,
            format="%.2f",
            help="Nilai pendapatan rumah tangga sesuai format dataset."
        )

    with col2:
        input_data["psize"] = st.number_input(
            "Ukuran Rombongan (psize)",
            min_value=1,
            value=1,
            step=1,
            format="%d",
            help="Jumlah anggota dalam rombongan."
        )

    st.divider()

    # ---------------------------------
    # B. ATRIBUT ALTERNATIF TRANSPORTASI
    # ---------------------------------

    st.markdown("### B. Atribut Alternatif Transportasi")

    st.caption(
        "Setiap alternatif memiliki empat atribut: waktu tempuh terminal, "
        "biaya dalam kendaraan, waktu dalam kendaraan, dan biaya gabungan."
    )

    for mode in range(1, 5):

        st.markdown(f"#### Alternatif Transportasi {mode}")

        col1, col2 = st.columns(2)

        with col1:
            input_data[f"ttme_mode_{mode}"] = st.number_input(
                f"Waktu Tempuh Terminal (ttme) - Alternatif {mode}",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=f"ttme_{mode}",
                help="Waktu tempuh menuju terminal."
            )

            input_data[f"invt_mode_{mode}"] = st.number_input(
                f"Waktu Dalam Kendaraan (invt) - Alternatif {mode}",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=f"invt_{mode}",
                help="Waktu yang dihabiskan di dalam kendaraan."
            )

        with col2:
            input_data[f"invc_mode_{mode}"] = st.number_input(
                f"Biaya Dalam Kendaraan (invc) - Alternatif {mode}",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=f"invc_{mode}",
                help="Biaya yang dikeluarkan dalam kendaraan."
            )

            input_data[f"gc_mode_{mode}"] = st.number_input(
                f"Biaya Gabungan (gc) - Alternatif {mode}",
                value=0.0,
                step=0.1,
                format="%.2f",
                key=f"gc_{mode}",
                help="Generalized cost atau biaya gabungan."
            )

    st.divider()

    submitted = st.form_submit_button(
        "Prediksi Pilihan Transportasi",
        use_container_width=True
    )


# =====================================
# HASIL PREDIKSI
# =====================================

if submitted:

    # Memastikan urutan fitur sesuai model
    input_df = pd.DataFrame(
        [[input_data[col] for col in features]],
        columns=features
    )

    prediction = model.predict(input_df)[0]
    probabilities = model.predict_proba(input_df)[0]

    st.divider()
    st.subheader("Hasil Prediksi")

    st.success(
        f"Kode moda transportasi yang diprediksi: {int(prediction)}"
    )

    # Tabel probabilitas
    probability_df = pd.DataFrame({
        "Kode Moda": classes,
        "Probabilitas": probabilities
    })

    probability_df["Persentase"] = (
        probability_df["Probabilitas"] * 100
    )

    st.markdown("### Probabilitas Setiap Alternatif")

    st.dataframe(
        probability_df[["Kode Moda", "Persentase"]].style.format({
            "Persentase": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Grafik probabilitas
    st.markdown("### Grafik Probabilitas")

    chart_df = probability_df.set_index("Kode Moda")

    st.bar_chart(chart_df["Probabilitas"])

    st.info(
        "Prediksi menunjukkan kelas dengan probabilitas tertinggi "
        "berdasarkan model Random Forest."
    )


# =====================================
# FOOTER
# =====================================

st.divider()

st.caption(
    "Mode Choice Prediction | Artificial Intelligence | "
    "Random Forest Classification"
)
