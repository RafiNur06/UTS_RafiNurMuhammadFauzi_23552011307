
import pandas as pd


def feature_engineering(df):

    # =====================================
    # 1. VALIDASI STRUKTUR DATA
    # =====================================

    required_columns = [
        'individual', 'mode', 'choice',
        'ttme', 'invc', 'invt', 'gc',
        'hinc', 'psize'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Kolom tidak ditemukan: {missing_columns}"
        )

    data = df[required_columns].copy()

    if data.isnull().any().any():
        raise ValueError("Dataset memiliki missing value.")

    # =====================================
    # 2. VALIDASI DATA INDIVIDU
    # =====================================

    # Setiap individu harus memiliki 4 alternatif
    jumlah_mode = data.groupby(
        'individual'
    )['mode'].nunique()

    if not (jumlah_mode == 4).all():
        raise ValueError(
            "Ada individu yang tidak memiliki 4 alternatif moda."
        )

    # Memastikan tidak ada kombinasi individu dan moda duplikat
    if data.duplicated(
        subset=['individual', 'mode']
    ).any():
        raise ValueError(
            "Terdapat kombinasi individu dan moda duplikat."
        )

    # Memastikan setiap individu memiliki tepat satu pilihan
    jumlah_pilihan = data.groupby(
        'individual'
    )['choice'].sum()

    if not (jumlah_pilihan == 1).all():
        raise ValueError(
            "Setiap individu harus memiliki tepat satu choice = 1."
        )

    # =====================================
    # 3. MEMERIKSA KONSISTENSI DATA
    # =====================================

    for col in ['hinc', 'psize']:
        konsistensi = data.groupby(
            'individual'
        )[col].nunique()

        if not (konsistensi == 1).all():
            raise ValueError(
                f"Nilai {col} tidak konsisten."
            )

    # =====================================
    # 4. MEMBUAT TARGET
    # =====================================

    target = (
        data[data['choice'] == 1]
        [['individual', 'mode']]
        .rename(columns={'mode': 'target'})
    )

    # =====================================
    # 5. MENGUBAH LONG KE WIDE FORMAT
    # =====================================

    # Fitur individu
    individu = data.groupby(
        'individual'
    ).agg({
        'hinc': 'first',
        'psize': 'first'
    })

    # Fitur masing-masing moda
    atribut = ['ttme', 'invc', 'invt', 'gc']

    wide_data = []

    for atribut_name in atribut:

        pivot = data.pivot(
            index='individual',
            columns='mode',
            values=atribut_name
        )

        pivot.columns = [
            f'{atribut_name}_mode_{int(mode)}'
            for mode in pivot.columns
        ]

        wide_data.append(pivot)

    # Menggabungkan seluruh fitur
    X = pd.concat(
        [individu] + wide_data,
        axis=1
    )

    # =====================================
    # 6. MENGGABUNGKAN TARGET
    # =====================================

    hasil = X.join(
        target.set_index('individual')
    )

    hasil = hasil.reset_index()

    # =====================================
    # 7. VALIDASI HASIL
    # =====================================

    if hasil.isnull().any().any():
        raise ValueError(
            "Hasil transformasi memiliki missing value."
        )

    if hasil['individual'].duplicated().any():
        raise ValueError(
            "Terdapat individu duplikat setelah transformasi."
        )

    return hasil
