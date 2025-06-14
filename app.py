import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib
import os

st.set_page_config(layout="wide")

st.title("Vehicle Emission Prediction Dashboard")
st.markdown("---")

# --- GLOBAL VARIABLES & PATHS FOR PKL FILES ---
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODERS_PATH = 'label_encoders.pkl'
MODEL_PATH_PREFIX = 'model_'
DATA_FILE_PATH = 'emission.csv'

# --- CACHED FUNCTIONS FOR LOADING RESOURCES ---

@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan di: {path}. Harap pastikan file ada di repositori.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.stop()

### FUNGSI INI DIPASTIKAN ADA DAN BENAR ###
@st.cache_resource
def load_trained_model(model_name, model_prefix_path):
    model_path = f"{model_prefix_path}{model_name.replace(' ', '_')}.pkl"
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
        
@st.cache_resource
def load_preprocessors_and_available_models(scaler_path, encoders_path, model_prefix):
    loaded_scaler = None
    loaded_label_encoders = None
    available_models_list = []

    try:
        loaded_scaler = joblib.load(scaler_path)
        loaded_label_encoders = joblib.load(encoders_path)
        
        for f in os.listdir('.'):
            if f.startswith(model_prefix) and f.endswith('.pkl'):
                available_models_list.append(f.replace(model_prefix, '').replace('.pkl', ''))
        
        if not available_models_list:
            st.warning("Tidak ada file model .pkl yang ditemukan di repositori. Pastikan Anda telah melatih model secara offline dan mengunggahnya.")

        return loaded_scaler, loaded_label_encoders, available_models_list
    except FileNotFoundError:
        st.error("File preprocessor (.pkl) tidak ditemukan. Harap pastikan Anda telah melatih model secara offline dan mengunggah file .pkl ke repositori.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat sumber daya: {e}. Pastikan file .pkl tidak rusak atau ada masalah kompatibilitas (periksa log untuk InconsistentVersionWarning).")
        st.stop()

# --- Load Preprocessors and Available Models (Dilakukan sekali di awal aplikasi) ---
scaler, encoders, available_models = load_preprocessors_and_available_models(SCALER_PATH, LABEL_ENCODERS_PATH, MODEL_PATH_PREFIX)

# Variabel ini akan menyimpan urutan kolom X_train setelah preprocessing
X_train_cols_order = [] 

# --- 1. Data Loading & Preprocessing ---
st.header("1. Data Loading & Preprocessing")

df = load_data(DATA_FILE_PATH)
df_original = df.copy()

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
        max_value=min(25, df_original['Make'].nunique()),
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


# --- Bagian Data Preprocessing (untuk mendapatkan X_train_cols_order) ---
st.subheader("Detail Preprocessing Data (Berdasarkan model yang dilatih offline)")

df_processed = df.copy()

columns_to_drop = ['Model', 'Make']
df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], errors='ignore')
st.write(f"Kolom yang dihapus dari data mentah: {columns_to_drop}")

X_temp = df_processed.drop(columns=[target_column_name])
y_temp = df_processed[target_column_name]

# Scaling Fitur (MinMaxScaler)
numerical_cols_temp = X_temp.select_dtypes(include=np.number).columns
if scaler and len(numerical_cols_temp) > 0:
    X_temp[numerical_cols_temp] = scaler.transform(X_temp[numerical_cols_temp])
    st.write("Fitur numerik diskalakan menggunakan MinMaxScaler yang dimuat.")
else:
    st.warning("Scaler tidak valid atau tidak ada kolom numerik untuk diskalakan.")

# Outlier Cleansing (IQR)
st.write("\nMelakukan Pembersihan Outlier (IQR)...")
original_rows_count = X_temp.shape[0]
if not numerical_cols_temp.empty:
    for col in numerical_cols_temp:
        Q1 = X_temp[col].quantile(0.25)
        Q3 = X_temp[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        filter_mask = (X_temp[col] >= lower_bound) & (X_temp[col] <= upper_bound)
        X_temp = X_temp[filter_mask]
        y_temp = y_temp.loc[X_temp.index]
    st.write(f"Setelah pembersihan outlier (IQR): {original_rows_count} -> {X_temp.shape[0]} baris.")

# Label Encoding
st.write("\nMelakukan Label Encoding...")
categorical_cols_for_le = ['Fuel Type', 'Transmission', 'Vehicle Class']
available_cat_cols_for_le = [col for col in categorical_cols_for_le if col in X_temp.columns and X_temp[col].dtype == 'object']
if encoders and available_cat_cols_for_le:
    for col in available_cat_cols_for_le:
        if X_temp[col].isnull().any():
            X_temp[col] = X_temp[col].fillna('Missing')
        X_temp[col] = encoders[col].transform(X_temp[col])
    st.write(f"Label Encoding berhasil diterapkan pada {available_cat_cols_for_le}.")

# Final data cleaning
X_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
y_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
X_temp.dropna(inplace=True)
y_temp = y_temp.loc[X_temp.index]

X_final = X_temp
y_final = y_temp

# Train-Test Split (untuk mendapatkan X_train_cols_order dan data evaluasi)
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
X_train_cols_order = X_train.columns.tolist() # Simpan urutan kolom untuk prediksi interaktif

st.success(f"Data berhasil diproses dan dibagi menjadi data latih dan data uji.")
st.markdown("---")

# --- 3. Model Evaluation (Sekarang hanya Muat Model) ---
st.header("3. Evaluasi Model (Menggunakan Model yang Sudah Terlatih)")

if available_models:
    model_choice_eval = st.selectbox(
        "Pilih Model untuk Evaluasi:",
        options=available_models,
        key='eval_model_select'
    )

    ### PEMANGGILAN FUNGSI DIPASTIKAN BENAR ###
    model_for_evaluation = load_trained_model(model_choice_eval, MODEL_PATH_PREFIX)
    
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

        if model_choice_eval == 'XGBoost':
            st.subheader("Pentingnya Fitur (XGBoost)")
            if hasattr(model_for_evaluation, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': X_train_cols_order,
                    'Importance': model_for_evaluation.feature_importances_
                }).sort_values(by='Importance', ascending=False)

                fig_feature_imp, ax_feature_imp = plt.subplots(figsize=(12, 7))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax_feature_imp)
                ax_feature_imp.set_title("15 Fitur Terpenting (XGBoost)")
                st.pyplot(fig_feature_imp)
else:
    st.info("Tidak ada model yang tersedia untuk evaluasi. Harap pastikan file .pkl sudah terunggah di repositori.")


### BAGIAN 5 INI DIUBAH TOTAL MENJADI LEBIH RAMAH PENGGUNA ###
# --- 5. Buat Prediksi Baru (Interaktif) ---
st.markdown("---")
st.header("5. Buat Prediksi Baru")

# Periksa apakah semua komponen siap untuk prediksi
if scaler is not None and encoders is not None and available_models and X_train_cols_order:
    model_to_predict_with = st.selectbox(
        "Pilih Model untuk Prediksi Interaktif:",
        options=available_models,
        key='predict_model_select'
    )
    
    ### PEMANGGILAN FUNGSI DIPASTIKAN BENAR ###
    loaded_model_predict = load_trained_model(model_to_predict_with, MODEL_PATH_PREFIX)

    if loaded_model_predict:
        st.write("Pilih kombinasi kendaraan untuk mendapatkan prediksi emisi CO2:")

        # Menggunakan Dependent Dropdowns
        col1, col2 = st.columns(2)
        
        # Ambil daftar unik dari DataFrame asli yang sudah dimuat di awal
        makes = sorted(df_original['Make'].unique())
        selected_make = col1.selectbox("Pilih Merek Mobil:", options=makes, key='pred_make')

        # Filter model berdasarkan merek yang dipilih
        models = sorted(df_original[df_original['Make'] == selected_make]['Model'].unique())
        selected_model = col2.selectbox("Pilih Model Mobil:", options=models, key='pred_model')
        
        df_filtered_options = df_original[(df_original['Make'] == selected_make) & (df_original['Model'] == selected_model)]

        col3, col4 = st.columns(2)

        transmissions = sorted(df_filtered_options['Transmission'].unique())
        selected_transmission = col3.selectbox("Pilih Transmisi:", options=transmissions, key='pred_transmission')

        fuel_types = sorted(df_filtered_options[df_filtered_options['Transmission'] == selected_transmission]['Fuel Type'].unique())
        selected_fuel_type = col4.selectbox("Pilih Tipe Bahan Bakar:", options=fuel_types, key='pred_fuel')

        if st.button("Dapatkan Prediksi"):
            # Cari baris yang cocok di DataFrame asli
            input_row = df_original[
                (df_original['Make'] == selected_make) &
                (df_original['Model'] == selected_model) &
                (df_original['Transmission'] == selected_transmission) &
                (df_original['Fuel Type'] == selected_fuel_type)
            ]

            if not input_row.empty:
                # Ambil baris data pertama yang cocok
                input_features_raw = input_row.iloc[[0]]
                actual_emission = input_features_raw[target_column_name].values[0]
                
                input_df_processed = input_features_raw.copy()

                try:
                    # --- Replikasi Preprocessing untuk Input yang Dicari ---

                    # 1. Label Encoding menggunakan encoder yang sudah dimuat
                    for col_name, encoder_obj in encoders.items():
                        if col_name in input_df_processed.columns:
                            input_df_processed[col_name] = encoder_obj.transform(input_df_processed[col_name])

                    # 2. Scaling Fitur Numerik menggunakan scaler yang sudah dimuat
                    numerical_cols_for_input_scaling = [col for col in numerical_cols_temp if col in input_df_processed.columns]
                    if scaler and numerical_cols_for_input_scaling:
                        input_df_processed[numerical_cols_for_input_scaling] = scaler.transform(input_df_processed[numerical_cols_for_input_scaling])

                    # 3. Pastikan urutan kolom SESUAI DENGAN X_train_cols_order
                    final_input_df = input_df_processed[X_train_cols_order]
                    
                    # Lakukan prediksi
                    prediction = loaded_model_predict.predict(final_input_df)
                    
                    # Tampilkan hasil
                    st.success("Prediksi Berhasil!")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Nilai Emisi Aktual (dari data)", f"{actual_emission:.2f} g/km")
                    res_col2.metric("Nilai Emisi Prediksi (oleh model)", f"{prediction[0]:.2f} g/km", delta=f"{prediction[0] - actual_emission:.2f}")

                except Exception as e:
                    st.error(f"Gagal membuat prediksi: {e}")
                    st.write("Pastikan preprocessor (.pkl) yang diunggah cocok dengan data yang digunakan.")

            else:
                st.error("Kombinasi kendaraan yang Anda pilih tidak ditemukan dalam dataset.")
    else:
        st.info("Silakan pilih model untuk prediksi interaktif.")
else:
    st.info("Model atau preprocessor belum siap untuk prediksi. Harap pastikan file .pkl sudah terunggah di repositori dan pemrosesan data utama selesai.")
