import streamlit as st
import pandas as pd
import numpy as np
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, MinMaxScaler # Perhatikan: Menggunakan MinMaxScaler sesuai final analisis
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
MODEL_PATH_PREFIX = 'model_' # Awalan untuk nama file model (misal: model_XGBoost.pkl)
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

@st.cache_resource
def load_preprocessors(scaler_path, encoders_path):
    try:
        loaded_scaler = joblib.load(scaler_path)
        loaded_label_encoders = joblib.load(encoders_path)
        return loaded_scaler, loaded_label_encoders
    except FileNotFoundError:
        st.error("File preprocessor (scaler.pkl atau label_encoders.pkl) tidak ditemukan. Harap pastikan Anda telah melatih model secara offline dan mengunggah file .pkl ke repositori.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat preprocessor: {e}. Pastikan file .pkl tidak rusak.")
        st.stop()

@st.cache_resource
def load_trained_model(model_name):
    model_path = f"{MODEL_PATH_PREFIX}{model_name.replace(' ', '_')}.pkl" # Menangani spasi di nama model
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            st.error(f"Model '{model_name}' tidak ditemukan di {model_path}. Harap pastikan Anda telah melatih model ini secara offline dan mengunggah file .pkl ke repositori.")
            st.stop() # Hentikan eksekusi jika model tidak ditemukan
    except Exception as e:
        st.error(f"Gagal memuat model '{model_name}': {e}. Pastikan file .pkl tidak rusak.")
        st.stop()

# --- 1. Data Loading & Preprocessing ---
st.header("1. Data Loading & Preprocessing")

df = load_data(DATA_FILE_PATH)
df_original = df.copy() # Untuk EDA yang menggunakan data asli

# --- Pilihan Kolom Target (tetap sama) ---
target_column_name = 'CO2 Emissions(g/km)' 
if target_column_name not in df.columns:
    st.error(f"Kolom target default '{target_column_name}' tidak ditemukan. Harap periksa dataset.")
    st.stop()
else:
    st.info(f"Menggunakan '{target_column_name}' sebagai kolom target.")

# --- Bagian Data Preprocessing (untuk mendapatkan X_train_cols_order) ---
# Bagian ini HARUS mereplikasi persis alur preprocessing dari notebook Anda
# Termasuk urutan MinMaxScaling sebelum Outlier Cleansing jika itu yang Anda lakukan
# Namun, scaler dan encoders yang digunakan di sini adalah yang DI-LOAD dari PKL, BUKAN di-fit ulang
st.subheader("Detail Preprocessing Data (Berdasarkan model yang dilatih offline)")

# Load preprocessors saat aplikasi dimulai
scaler, encoders = load_preprocessors(SCALER_PATH, LABEL_ENCODERS_PATH)
st.success("Scaler dan LabelEncoders berhasil dimuat.")

# Replikasi preprocessing untuk mendapatkan X_train_cols_order dan Y
# Ini untuk memastikan konsistensi kolom dan melakukan split untuk evaluasi
df_processed = df.copy()

# 1. Penghapusan Kolom Awal (sama seperti di notebook)
columns_to_drop = [
    'Model', 'Make', 'Fuel Consumption (City (L/100 km))',
    'Fuel Consumption(Hwy (L/100 km))', 'Fuel Consumption(Comb (L/100 km))'
]
df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], errors='ignore')

# 2. Pemisahan X dan y (sementara, sebelum outlier/scaling/encoding)
X_temp = df_processed.drop(columns=[target_column_name])
y_temp = df_processed[target_column_name]

# 3. Scaling Fitur (MinMaxScaler) - REPLIKASI URUTAN DARI NOTEBOOK ANDA
numerical_cols_temp = X_temp.select_dtypes(include=np.number).columns
if scaler and len(numerical_cols_temp) > 0:
    X_temp[numerical_cols_temp] = scaler.transform(X_temp[numerical_cols_temp])
    st.write("Fitur numerik diskalakan menggunakan MinMaxScaler yang dimuat.")

# 4. Outlier Cleansing (IQR) - REPLIKASI URUTAN DARI NOTEBOOK ANDA (setelah scaling)
# Penting: Outlier cleansing harus menggunakan batas yang DITENTUKAN dari data training
# Namun, dalam konteks deployment, kita tidak memiliki akses ke batas Q1/Q3 training secara langsung tanpa menyimpannya.
# Jika IQR Anda menghapus baris, maka jumlah baris X dan y akan berubah.
# Untuk tujuan deployment yang stabil, *idealnya* outlier cleansing adalah bagian dari proses data training
# dan baris yang dihapus sudah final.

# Jika IQR menghapus baris, maka X_train_cols_order harus berasal dari X_train final
# yang sudah melalui IQR. Ini yang paling rumit jika IQR tidak disimpan.
# Solusi paling robust: Lakukan IQR di notebook, dapatkan index baris yang tersisa,
# dan gunakan index itu untuk memfilter X dan y di streamlit.
# Namun, saya akan mengikuti logika yang memodifikasi X_temp dan y_temp.
original_rows_count = X_temp.shape[0]
if numerical_cols_temp.empty:
    st.info("Tidak ada kolom numerik untuk pembersihan outlier.")
else:
    # Memfilter berdasarkan outlier pada data yang sudah diskalakan
    for col in numerical_cols_temp:
        # PENTING: Untuk deployment, batas Q1/Q3 idealnya diambil dari data training
        # dan disimpan/dimuat, bukan dihitung ulang dari df_processed.
        # Namun, karena notebook Anda tidak menyimpan batas IQR, kita hitung ulang dari df_processed
        # Ini bisa menghasilkan sedikit perbedaan jika distribusi data deployment tidak persis sama
        # dengan data yang melatih model.
        Q1 = X_temp[col].quantile(0.25)
        Q3 = X_temp[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        X_temp = X_temp[(X_temp[col] >= lower_bound) & (X_temp[col] <= upper_bound)]
        y_temp = y_temp.loc[X_temp.index] # Pastikan y selaras
    st.write(f"Setelah pembersihan outlier (IQR): {original_rows_count} -> {X_temp.shape[0]} baris.")


# 5. Label Encoding (setelah outlier cleansing)
# Gunakan encoders yang sudah di-load
categorical_cols_for_le = ['Fuel Type', 'Transmission', 'Vehicle Class']
available_cat_cols_for_le = [col for col in categorical_cols_for_le if col in X_temp.columns and X_temp[col].dtype == 'object']

if available_cat_cols_for_le:
    st.write(f"Menerapkan Label Encoding pada: {available_cat_cols_for_le}")
    for col in available_cat_cols_for_le:
        if X_temp[col].isnull().any():
            X_temp[col] = X_temp[col].fillna('Missing')
        # Gunakan encoder yang dimuat untuk transform
        X_temp[col] = encoders[col].transform(X_temp[col])
    st.write("Label Encoding berhasil diterapkan.")


# Validasi kolom target (tetap sama)
if not pd.api.types.is_numeric_dtype(y_temp):
    st.error(f"Kolom target '{target_column_name}' mengandung nilai non-numerik. Pastikan itu numerik untuk regresi.")
    st.stop()

# Handle inf/nan after all preprocessing (final cleanup)
X_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
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

model_choice_eval = st.selectbox(
    "Pilih Model untuk Evaluasi:",
    ('XGBoost', 'Lasso Regression', 'Linear Regression', 'Support Vector Regressor (SVR)'),
    key='eval_model_select'
)

# Muat model yang dipilih
model_for_evaluation = load_trained_model(model_choice_eval)

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
        # Karena X_final adalah DataFrame yang sudah diproses dan akan menjadi X_train
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


# --- 5. Make a New Prediction (Interaktif) ---
st.markdown("---")
st.header("5. Buat Prediksi Baru")

if scaler and encoders and available_models and X_train_cols_order:
    model_to_predict_with = st.selectbox(
        "Pilih Model untuk Prediksi Interaktif:",
        options=available_models,
        key='predict_model_select'
    )
    loaded_model_predict = load_trained_model(model_to_predict_with)

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
            # Fuel Consumption(Comb (L/100km)) dihilangkan karena didrop di preprocessing
            # Jika Anda ingin ini sebagai input, HAPUS dari 'columns_to_drop' di bagian preprocessing.
            # input_fuel_comb = st.number_input('Konsumsi Bahan Bakar (L/100km)', min_value=5.0, max_value=25.0, value=10.0, step=0.1, key='input_fuel_comb_pred')

        if st.button("Dapatkan Prediksi"):
            try:
                # Membuat dictionary input mentah dari user
                raw_input_data = {
                    'Engine Size(L)': input_engine_size,
                    'Cylinders': input_cylinders,
                    'Transmission': input_transmission,
                    'Fuel Type': input_fuel_type,
                    'Vehicle Class': input_vehicle_class,
                }
                input_df_raw = pd.DataFrame([raw_input_data])
                
                # --- Replikasi Preprocessing untuk Input Prediksi ---

                # 1. Scaling Fitur (MinMaxScaler) pada input MENTAH (sebelum outlier)
                # Pastikan input_df_raw memiliki kolom numerik yang sama dengan yang diskalakan model
                numerical_cols_input = input_df_raw.select_dtypes(include=np.number).columns
                if scaler and len(numerical_cols_input) > 0:
                    input_df_raw[numerical_cols_input] = scaler.transform(input_df_raw[numerical_cols_input])
                
                # 2. Outlier Cleansing (IQR) - Replikasikan logika ini untuk input tunggal
                # Ini adalah bagian yang tricky. Untuk input tunggal, Anda tidak bisa
                # menghapus baris. Anda harus memutuskan bagaimana menangani nilai yang akan menjadi outlier.
                # Opsi: Jika nilai berada di luar batas IQR (yang sudah dihitung dari data training),
                # Anda bisa mengklipnya ke batas tersebut atau memberi peringatan.
                # Untuk kesederhanaan, kita akan menganggap input user ada dalam rentang wajar setelah scaling,
                # karena IQR di data training menghapus baris, bukan mengubah nilai.
                # Jika Anda benar-benar ingin meniru, Anda harus menyimpan batas IQR (Q1, Q3) dari data training.
                st.warning("Penting: Pembersihan outlier (IQR) pada input tunggal tidak menghapus baris seperti pada data training. Diasumsikan input Anda berada dalam rentang yang wajar atau batas outlier akan diterapkan secara implisit oleh model setelah scaling.")
                
                # 3. Label Encoding
                for col_name, encoder in encoders.items():
                    if col_name in input_df_raw.columns:
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
                        final_input_df[col] = 0 # Default untuk fitur yang tidak diinput
                
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
    st.info("Model atau preprocessor belum siap untuk prediksi. Harap pastikan file .pkl sudah terunggah di repositori.")
