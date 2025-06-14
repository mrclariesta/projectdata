import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # Pastikan MinMaxScaler diimpor
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib
import os # Pastikan ini diimpor untuk os.listdir

st.set_page_config(layout="wide")

st.title("Vehicle Emission Prediction Dashboard")
st.markdown("---")

# --- GLOBAL VARIABLES & PATHS FOR PKL FILES ---
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODERS_PATH = 'label_encoders.pkl'
MODEL_PATH_PREFIX = 'model_' # Awalan untuk nama file model (misal: model_XGBoost.pkl)
DATA_FILE_PATH = 'emission.csv' # Nama file CSV Anda

# --- CACHED FUNCTIONS FOR LOADING RESOURCES ---

@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan di: {path}. Harap pastikan file ada di repositori.")
        st.stop() # Hentikan eksekusi jika file tidak ditemukan
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.stop() # Hentikan eksekusi jika ada error lain

@st.cache_resource
def load_preprocessors_and_available_models(scaler_path, encoders_path, model_prefix):
    loaded_scaler = None
    loaded_label_encoders = None
    available_models_list = []

    try:
        # Coba muat scaler dan encoders
        loaded_scaler = joblib.load(scaler_path)
        loaded_label_encoders = joblib.load(encoders_path)
        
        # Cari file model .pkl yang tersedia di direktori
        for f in os.listdir('.'):
            if f.startswith(model_prefix) and f.endswith('.pkl'):
                available_models_list.append(f.replace(model_prefix, '').replace('.pkl', ''))
        
        if not available_models_list:
            st.warning("Tidak ada file model .pkl yang ditemukan di repositori. Pastikan Anda telah melatih model secara offline dan mengunggahnya.")

        return loaded_scaler, loaded_label_encoders, available_models_list
    except FileNotFoundError:
        st.error("File preprocessor (.pkl) tidak ditemukan. Harap pastikan Anda telah melatih model secara offline dan mengunggah file .pkl ke repositori.")
        st.stop() # Hentikan eksekusi jika file tidak ditemukan
    except Exception as e:
        st.error(f"Error memuat sumber daya: {e}. Pastikan file .pkl tidak rusak atau ada masalah kompatibilitas (periksa log untuk InconsistentVersionWarning).")
        st.stop() # Hentikan eksekusi jika ada error lain

@st.cache_resource
def load_trained_model(model_name, model_prefix_path): # Tambahkan model_prefix_path sebagai argumen
    model_path = f"{model_prefix_path}{model_name.replace(' ', '_')}.pkl" # Menggunakan model_prefix_path yang diteruskan
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            st.error(f"Model '{model_name}' tidak ditemukan di {model_path}. Harap pastikan Anda telah melatih model ini secara offline dan mengunggah file .pkl ke repositori.")
            st.stop()
    except Exception as e:
        st.error(f"Gagal memuat model '{model_name}': {e}. Pastikan file .pkl tidak rusak atau terjadi masalah kompatibilitas.")
        st.stop()

# --- Load Preprocessors and Available Models (Dilakukan sekali di awal aplikasi) ---
# Variabel 'scaler', 'encoders', dan 'available_models' akan tersedia secara global setelah ini.
scaler, encoders, available_models = load_preprocessors_and_available_models(SCALER_PATH, LABEL_ENCODERS_PATH, MODEL_PATH_PREFIX)

# Variabel ini akan menyimpan urutan kolom X_train setelah preprocessing
# Penting untuk prediksi interaktif
X_train_cols_order = [] 

# --- 1. Data Loading & Preprocessing ---
st.header("1. Data Loading & Preprocessing")

df = load_data(DATA_FILE_PATH)
df_original = df.copy() # Untuk EDA yang menggunakan data asli

# --- Pilihan Kolom Target ---
target_column_name = 'CO2 Emissions(g/km)' 
if target_column_name not in df.columns:
    st.error(f"Kolom target default '{target_column_name}' tidak ditemukan. Harap periksa dataset.")
    st.stop()
else:
    st.info(f"Menggunakan '{target_column_name}' sebagai kolom target.")

# --- Initial Data Insights: Perbandingan CO2 Emissions(g/km) dengan Merek Umum ---
if 'Make' in df.columns and target_column_name in df.columns:
    st.subheader(f"Perbandingan {target_column_name} Rata-rata per Merek Umum")

    num_common_makes = st.slider(
        "Pilih jumlah merek mobil paling umum yang akan ditampilkan:",
        min_value=5,
        max_value=min(25, df['Make'].nunique()),
        value=10,
        step=1,
        key='make_slider'
    )

    common_makes = df_original['Make'].value_counts().nlargest(num_common_makes).index.tolist()

    df_common_makes_filtered = df_original[df_original['Make'].isin(common_makes)]

    make_avg_co2_filtered = df_common_makes_filtered.groupby('Make')[target_column_name].mean().reset_index()
    make_avg_co2_filtered = make_avg_co2_filtered.sort_values(by=target_column_name, ascending=False)

    fig_make_co2, ax_make_co2 = plt.subplots(figsize=(14, 7))
    sns.barplot(x='Make', y=target_column_name, data=make_avg_co2_filtered, ax=ax_make_co2)
    ax_make_co2.set_title(f"Rata-rata {target_column_name} per {num_common_makes} Merek Mobil Paling Umum")
    ax_make_co2.set_xlabel("Merek Mobil ('Make')")
    ax_make_co2.set_ylabel(target_column_name)
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_make_co2)
    st.markdown("---")
else:
    st.warning("Kolom 'Make' atau kolom target tidak ditemukan untuk visualisasi 'Make' vs CO2.")

# --- New Plot: Box Plot / Violin Plot untuk Fitur Kategorikal Asli vs. Target ---
st.subheader(f"Distribusi '{target_column_name}' per Fitur Kategorikal")

original_categorical_features = ['Fuel Type', 'Transmission', 'Vehicle Class']

available_original_categorical_features = [
    col for col in original_categorical_features
    if col in df_original.columns and df_original[col].dtype == 'object'
]

if available_original_categorical_features and target_column_name in df_original.columns:
    selected_cat_feature = st.selectbox(
        "Pilih Fitur Kategorikal:",
        options=available_original_categorical_features,
        key='cat_feature_select'
    )
    if selected_cat_feature:
        plot_type = st.radio("Pilih Jenis Plot:", ('Box Plot', 'Violin Plot'), key='cat_plot_type')
        fig_cat_target, ax_cat_target = plt.subplots(figsize=(12, 7))
        
        if plot_type == 'Box Plot':
            sns.boxplot(x=selected_cat_feature, y=target_column_name, data=df_original, ax=ax_cat_target)
        else: # Violin Plot
            sns.violinplot(x=selected_cat_feature, y=target_column_name, data=df_original, ax=ax_cat_target)
        
        ax_cat_target.set_title(f"{plot_type}: {target_column_name} per {selected_cat_feature}")
        ax_cat_target.set_xlabel(selected_cat_feature)
        ax_cat_target.set_ylabel(target_column_name)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig_cat_target)
    st.markdown("---")
else:
    st.info("Tidak ada fitur kategorikal asli yang ditemukan atau kolom target hilang untuk visualisasi kategori vs target.")


# --- Bagian Data Preprocessing (untuk mendapatkan X_train_cols_order dan data yang siap untuk evaluasi) ---
# Bagian ini HARUS mereplikasi persis alur preprocessing dari notebook Anda
# Termasuk urutan MinMaxScaler sebelum Outlier Cleansing
st.subheader("Detail Preprocessing Data (Berdasarkan model yang dilatih offline)")

# Replikasi preprocessing untuk mendapatkan X_final dan y_final yang siap untuk split dan visualisasi
df_processed = df.copy()

# 1. Penghapusan Kolom Awal (sesuai notebook, HANYA model, make)
# Karena kolom Fuel Consumption digunakan sebagai fitur, JANGAN HAPUS DI SINI
columns_to_drop = [
    'Model', 'Make' # Fuel Consumption columns DIJAGA agar jadi fitur
]
df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], errors='ignore')
st.write(f"Kolom yang dihapus dari data mentah: {columns_to_drop}")

# 2. Pemisahan X dan y Awal
X_temp = df_processed.drop(columns=[target_column_name])
y_temp = df_processed[target_column_name]

# --- Diagnostik scaler ---
st.write(f"Scaler (setelah dimuat): {type(scaler).__name__}")
if hasattr(scaler, 'min_') and hasattr(scaler, 'scale_'):
    st.write(f"Scaler (MinMaxScaler) sudah di-fit. Data train fit range: {scaler.min_[0]:.2f} - {scaler.scale_[0] + scaler.min_[0]:.2f} (contoh fitur pertama).")
else:
    st.warning("Peringatan: Scaler (MinMaxScaler) dimuat tetapi tampaknya belum di-fit dengan benar (atribut min_ atau scale_ tidak ada).")


# 3. Scaling Fitur (MinMaxScaler) - REPLIKASI URUTAN DARI NOTEBOOK ANDA (sebelum Outlier)
numerical_cols_temp = X_temp.select_dtypes(include=np.number).columns
if scaler and len(numerical_cols_temp) > 0:
    X_temp[numerical_cols_temp] = scaler.transform(X_temp[numerical_cols_temp])
    st.write("Fitur numerik diskalakan menggunakan MinMaxScaler yang dimuat.")
else:
    st.warning("Scaler tidak valid atau tidak ada kolom numerik untuk diskalakan.")


# 4. Outlier Cleansing (IQR) - REPLIKASI URUTAN DARI NOTEBOOK ANDA (setelah scaling)
st.write("\nMelakukan Pembersihan Outlier (IQR)...")
original_rows_count = X_temp.shape[0]
if numerical_cols_temp.empty:
    st.info("Tidak ada kolom numerik untuk pembersihan outlier.")
else:
    # Memfilter berdasarkan outlier pada data yang sudah diskalakan
    for col in numerical_cols_temp: # Iterasi hanya pada kolom numerik yang diskalakan
        Q1 = X_temp[col].quantile(0.25)
        Q3 = X_temp[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filter_mask = (X_temp[col] >= lower_bound) & (X_temp[col] <= upper_bound)
        X_temp = X_temp[filter_mask]
        y_temp = y_temp.loc[X_temp.index] # Selaraskan y
    st.write(f"Setelah pembersihan outlier (IQR): {original_rows_count} -> {X_temp.shape[0]} baris.")


# 5. Label Encoding (setelah outlier cleansing)
st.write("\nMelakukan Label Encoding...")
categorical_cols_for_le = ['Fuel Type', 'Transmission', 'Vehicle Class']
available_cat_cols_for_le = [col for col in categorical_cols_for_le if col in X_temp.columns and X_temp[col].dtype == 'object']

if encoders and available_cat_cols_for_le: # Pastikan encoders dimuat dan ada kolom kategorikal
    for col in available_cat_cols_for_le:
        if X_temp[col].isnull().any():
            X_temp[col] = X_temp[col].fillna('Missing')
        # Gunakan encoder yang dimuat untuk transform
        X_temp[col] = encoders[col].transform(X_temp[col])
    st.write(f"Label Encoding berhasil diterapkan pada {available_cat_cols_for_le}.")
elif not encoders:
    st.warning("Encoders (LabelEncoders) tidak dimuat. Label Encoding dilewati.")
else:
    st.info("Tidak ada kolom kategorikal yang tersedia untuk Label Encoding setelah pembersihan outlier.")


# Validasi kolom target (tetap sama)
if not pd.api.types.is_numeric_dtype(y_temp):
    st.error(f"Kolom target '{target_column_name}' mengandung nilai non-numerik. Pastikan itu numerik untuk regresi.")
    st.stop()

# Handle inf/nan after all preprocessing (final cleanup)
X_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
y_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_rows_X = X_temp.shape[0]
X_temp.dropna(inplace=True)
y_temp = y_temp.loc[X_temp.index]
if X_temp.shape[0] < initial_rows_X:
    st.warning(f"Menghapus {initial_rows_X - X_temp.shape[0]} baris mengandung NaN/Inf setelah preprocessing akhir.")

# X_final dan y_final untuk train-test split dan visualisasi
X_final = X_temp
y_final = y_temp

st.write("Bentuk X (fitur) setelah semua preprocessing:", X_final.shape)
st.write("Bentuk y (target) setelah semua preprocessing:", y_final.shape)
st.dataframe(X_final.head())

# Train-Test Split (untuk mendapatkan X_train_cols_order dan data evaluasi)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
X_train_cols_order = X_train.columns.tolist() # Simpan urutan kolom untuk prediksi interaktif

st.success(f"Data dibagi menjadi X_train ({X_train.shape}), X_test ({X_test.shape}), y_train ({y_train.shape}), y_test ({y_test.shape})")
st.write("X_train head:")
st.dataframe(X_train.head())

# --- Visualisasi EDA Tambahan (menggunakan data yang sudah diproses secara parsial jika perlu) ---
# Distribusi Emisi Karbon
st.subheader(f"Distribusi '{target_column_name}'")
fig_dist_co2, ax_dist_co2 = plt.subplots(figsize=(10, 6))
sns.histplot(y_final, kde=True, ax=ax_dist_co2, bins=30)
ax_dist_co2.set_title(f"Distribusi '{target_column_name}'")
ax_dist_co2.set_xlabel(target_column_name)
ax_dist_co2.set_ylabel("Frekuensi")
st.pyplot(fig_dist_co2)
st.markdown("---")

# Heatmap Korelasi
st.subheader("Heatmap Korelasi Antar Fitur dan Target")
df_for_corr = pd.concat([X_final, y_final], axis=1)
correlation_matrix = df_for_corr.corr(numeric_only=True)
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
ax_corr.set_title("Heatmap Korelasi Antar Fitur dan Target")
st.pyplot(fig_corr)
st.markdown("---")

# --- 3. Model Evaluation (Sekarang hanya Muat Model) ---
st.header("3. Evaluasi Model (Menggunakan Model yang Sudah Terlatih)")

if available_models: # Cek apakah ada model yang tersedia dari hasil load_preprocessors_and_available_models
    model_choice_eval = st.selectbox(
        "Pilih Model untuk Evaluasi:",
        options=available_models,
        key='eval_model_select'
    )

    model_for_evaluation = load_trained_model(model_choice_eval, MODEL_PATH_PREFIX) # Teruskan MODEL_PATH_PREFIX
    
    if model_for_evaluation:
        y_pred_eval = model_for_evaluation.predict(X_test)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred_eval),
            'r2': r2_score(y_test, y_pred_eval),
            'mae': mean_absolute_error(y_test, y_pred_eval),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_eval))
        }

        st.subheader(f"Metrik Evaluasi untuk {model_choice_eval}:")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.4f}")
        col2.metric("R-Squared (R2)", f"{metrics['r2']:.4f}")
        col3.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}")
        col4.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.4f}")
        st.markdown("---")

        # --- 4. Visualizations ---
        st.header("4. Visualisasi Hasil Model")

        st.subheader(f"Plot antara Fitur dan {target_column_name}")
        feature_to_plot = st.selectbox(
            "Pilih Fitur untuk di-plot terhadap target:",
            options=X_final.columns.tolist(), # Gunakan X_final yang sudah diproses
            key='feature_target_plot_select'
        )
        if feature_to_plot:
            fig_feature_target, ax_feature_target = plt.subplots(figsize=(10, 6))
            
            if pd.api.types.is_numeric_dtype(X_final[feature_to_plot]):
                sns.scatterplot(x=X_final[feature_to_plot], y=y_final, ax=ax_feature_target, alpha=0.6)
                ax_feature_target.set_title(f"Scatter Plot: {feature_to_plot} vs {target_column_name}")
            else:
                sns.boxplot(x=X_final[feature_to_plot], y=y_final, ax=ax_feature_target)
                ax_feature_target.set_title(f"Box Plot: {feature_to_plot} vs {target_column_name}")
            
            ax_feature_target.set_xlabel(feature_to_plot)
            ax_feature_target.set_ylabel(target_column_name)
            st.pyplot(fig_feature_target)
        st.markdown("---")


        st.subheader("Aktual vs. Prediksi")
        fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
        ax_scatter.scatter(y_test, y_pred_eval, alpha=0.7)
        ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax_scatter.set_xlabel(f"Nilai Aktual {target_column_name}")
        ax_scatter.set_ylabel(f"Nilai Prediksi {target_column_name}")
        ax_scatter.set_title(f"Nilai Aktual vs. Prediksi (Test Set) - {model_choice_eval}")
        st.pyplot(fig_scatter)

        st.subheader("Plot Residual")
        residuals = y_test - y_pred_eval
        fig_residuals, ax_residuals = plt.subplots(figsize=(10, 6))
        ax_residuals.scatter(y_pred_eval, residuals, alpha=0.7)
        ax_residuals.axhline(y=0, color='r', linestyle='--', lw=2)
        ax_residuals.set_xlabel(f"Nilai Prediksi {target_column_name}")
        ax_residuals.set_ylabel("Residual (Aktual - Prediksi)")
        ax_residuals.set_title(f"Plot Residual - {model_choice_eval}")
        st.pyplot(fig_residuals)

        st.subheader("Distribusi Residual")
        fig_hist_residuals, ax_hist_residuals = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, ax=ax_hist_residuals)
        ax_hist_residuals.set_xlabel("Residual")
        ax_hist_residuals.set_ylabel("Frekuensi")
        ax_hist_residuals.set_title(f"Distribusi Residual - {model_choice_eval}")
        st.pyplot(fig_hist_residuals)

        st.subheader("QQ-Plot Residual")
        fig_qq, ax_qq = plt.subplots(figsize=(8, 8))
        sm.qqplot(residuals, line='s', ax=ax_qq)
        ax_qq.set_title(f"QQ-Plot Residual - {model_choice_eval}")
        st.pyplot(fig_qq)
        st.markdown("---")

        if model_choice_eval == 'XGBoost':
            st.subheader("Pentingnya Fitur (XGBoost)")
            feature_names_for_importance = X_final.columns.tolist() 
            if hasattr(model_for_evaluation, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': feature_names_for_importance,
                    'Importance': model_for_evaluation.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                fig_feature_imp, ax_feature_imp = plt.subplots(figsize=(12, 7))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax_feature_imp)
                ax_feature_imp.set_title("15 Fitur Terpenting (XGBoost)")
                ax_feature_imp.set_xlabel("Skor Kepentingan")
                ax_feature_imp.set_ylabel("Fitur")
                st.pyplot(fig_feature_imp)
            else:
                st.info("Atribut feature_importances_ tidak ditemukan untuk model XGBoost ini.")
        else:
            st.info("Pentingnya fitur biasanya ditampilkan untuk model berbasis pohon seperti XGBoost.")

        st.subheader("Contoh Nilai Aktual vs. Prediksi (Test Set)")
        results_df = pd.DataFrame({
            'Aktual': y_test,
            'Prediksi': y_pred_eval,
            'Residual': residuals
        }).head(15)
        st.dataframe(results_df)

    else:
        st.warning("Model belum dimuat untuk evaluasi. Pilih model dari dropdown.")

else: # Jika available_models kosong (tidak ada model .pkl yang ditemukan)
    st.info("Tidak ada model yang tersedia untuk evaluasi. Harap pastikan file .pkl sudah terunggah di repositori.")


# --- 5. Make a New Prediction (Interaktif) ---
st.markdown("---")
st.header("5. Buat Prediksi Baru")

# Periksa apakah scaler, encoders, dan ada model yang tersedia untuk prediksi
if scaler is not None and encoders is not None and available_models and X_train_cols_order:
    model_to_predict_with = st.selectbox(
        "Pilih Model untuk Prediksi Interaktif:",
        options=available_models,
        key='predict_model_select'
    )
    loaded_model_predict = load_trained_model(model_to_predict_with, MODEL_PATH_PREFIX) # Teruskan MODEL_PATH_PREFIX

    if loaded_model_predict:
        st.write("Masukkan nilai fitur untuk mendapatkan prediksi emisi CO2:")

        col_input1, col_input2, col_input3 = st.columns(3)

        # Options for selectboxes from the original loaded df
        make_options = df_original['Make'].unique().tolist()
        transmission_options = df_original['Transmission'].unique().tolist()
        fuel_type_options = df_original['Fuel Type'].unique().tolist()
        vehicle_class_options = df_original['Vehicle Class'].unique().tolist()

        with col_input1:
            input_make = st.selectbox('Merk Mobil', options=make_options, key='input_make_pred')
            
            # Dynamic Model selection based on Make
            filtered_models = df_original[df_original['Make'] == input_make]['Model'].unique().tolist() if input_make else []
            input_model = st.selectbox('Tipe Mobil', options=filtered_models, key='input_model_pred')
            input_engine_size = st.number_input('Ukuran Mesin (L)', min_value=0.5, max_value=8.0, value=2.0, step=0.1, key='input_engine_size_pred')

        with col_input2:
            input_cylinders = st.number_input('Jumlah Silinder', min_value=3, max_value=12, value=4, step=1, key='input_cylinders_pred')
            input_transmission = st.selectbox('Transmisi', options=transmission_options, key='input_transmission_pred')
            input_fuel_type = st.selectbox('Tipe Bahan Bakar', options=fuel_type_options, key='input_fuel_type_pred')

        with col_input3:
            input_vehicle_class = st.selectbox('Kelas Kendaraan', options=vehicle_class_options, key='input_vehicle_class_pred')
            # Tambahkan input untuk kolom Fuel Consumption karena sekarang mereka adalah fitur
            input_fuel_city = st.number_input('Konsumsi BBM Kota (L/100km)', min_value=1.0, max_value=50.0, value=10.0, step=0.1, key='input_fuel_city_pred')
            input_fuel_hwy = st.number_input('Konsumsi BBM Tol (L/100km)', min_value=1.0, max_value=40.0, value=8.0, step=0.1, key='input_fuel_hwy_pred')
            input_fuel_comb_l = st.number_input('Konsumsi BBM Gabungan (L/100km)', min_value=1.0, max_value=45.0, value=9.0, step=0.1, key='input_fuel_comb_l_pred')
            input_fuel_comb_mpg = st.number_input('Konsumsi BBM Gabungan (mpg)', min_value=5.0, max_value=80.0, value=25.0, step=0.1, key='input_fuel_comb_mpg_pred')

        if st.button("Dapatkan Prediksi"):
            try:
                # Membuat dictionary input mentah dari user
                # Pastikan semua kunci ini sesuai dengan X_train_cols_order (setelah preprocessing)
                raw_input_data = {
                    'Engine Size(L)': input_engine_size,
                    'Cylinders': input_cylinders,
                    'Transmission': input_transmission,
                    'Fuel Type': input_fuel_type,
                    'Vehicle Class': input_vehicle_class,
                    'Fuel Consumption (City (L/100 km))': input_fuel_city,
                    'Fuel Consumption (Hwy (L/100 km))': input_fuel_hwy,
                    'Fuel Consumption (Comb (L/100 km))': input_fuel_comb_l,
                    'Fuel Consumption (Comb (mpg))': input_fuel_comb_mpg,
                }
                input_df_raw = pd.DataFrame([raw_input_data])
                
                # --- Replikasi Preprocessing untuk Input Prediksi ---

                # 1. Scaling Fitur (MinMaxScaler) pada input MENTAH (sebelum outlier)
                # Gunakan 'numerical_cols_for_scaler_fit' dari bagian preprocessing utama
                # untuk memastikan kolom yang sama diskalakan.
                numerical_cols_for_input_scaling = [col for col in numerical_cols_temp if col in input_df_raw.columns] # Pastikan hanya kolom numerik yang ada di input_df_raw
                if scaler and len(numerical_cols_for_input_scaling) > 0:
                    input_df_raw[numerical_cols_for_input_scaling] = scaler.transform(input_df_raw[numerical_cols_for_input_scaling])
                else:
                    st.warning("Input numerik tidak diskalakan karena scaler tidak valid atau tidak ada kolom numerik.")

                # 2. Outlier Cleansing (IQR) - Replikasikan logika ini untuk input tunggal
                st.info("Catatan: Pembersihan outlier (IQR) pada input tunggal tidak menghapus baris. Diasumsikan input dalam rentang wajar.")
                # Anda bisa menambahkan clipping di sini jika nilai input di luar batas Q1-1.5IQR atau Q3+1.5IQR.
                
                # 3. Label Encoding
                for col_name, encoder in encoders.items():
                    if col_name in input_df_raw.columns: # Hanya jika kolom ada di input
                        if input_df_raw[col_name].iloc[0] not in encoder.classes_:
                            st.error(f"Kategori '{input_df_raw[col_name].iloc[0]}' di kolom '{col_name}' tidak dikenali oleh model. Harap pilih dari daftar yang tersedia dari data pelatihan.")
                            st.stop()
                        input_df_raw[col_name] = encoder.transform(input_df_raw[col_name])
                
                # 4. Pastikan urutan kolom sesuai dengan X_train_cols_order
                final_input_df = pd.DataFrame(columns=X_train_cols_order)
                for col in X_train_cols_order:
                    if col in input_df_raw.columns:
                        final_input_df[col] = input_df_raw[col]
                    else:
                        # Mengisi kolom yang tidak diinput dengan 0 atau rata-rata dari training data
                        final_input_df[col] = 0 
                
                # Lakukan prediksi
                prediction = loaded_model_predict.predict(final_input_df)
                
                st.success("Prediksi Emisi CO2:")
                st.write(f"**{prediction[0]:.2f} g/km**")
                
            except Exception as e:
                st.error(f"Gagal membuat prediksi: {e}.")
                st.write("Pastikan semua input valid dan preprocessing benar.")
                st.write("Debug Info:")
                st.write(f"Input DataFrame (before final processing): {input_df_raw}")
                st.write(f"Expected X_train columns: {X_train_cols_order}")
    else:
        st.info("Silakan pilih model untuk prediksi interaktif.")
else:
    st.info("Model atau preprocessor belum siap untuk prediksi. Harap pastikan file .pkl sudah terunggah di repositori dan pemrosesan data utama selesai.")
