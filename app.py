import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import joblib
import os

st.set_page_config(layout="wide")

st.title("Vehicle Emission Prediction Dashboard")
st.markdown("---")

# --- GLOBAL VARIABLES & CACHED COMPONENTS ---
SCALER_PATH = 'scaler.pkl'
LABEL_ENCODERS_PATH = 'label_encoders.pkl'
MODEL_PATH_PREFIX = 'model_'

# --- Load Data ---
st.header("1. Data Loading & Preprocessing")

data_file_path = 'emission.csv' 

@st.cache_data 
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan di: {path}. Harap pastikan file ada di repositori atau ubah path.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.stop()

df = load_data(data_file_path)

# Initialize X_train_cols_order globally
X_train_cols_order = [] 

# --- Pilihan Kolom Target (tetap sama) ---
target_column_name = 'CO2 Emissions(g/km)' 
if target_column_name not in df.columns:
    st.error(f"Default target column '{target_column_name}' not found. Please select manually.")
    target_column_name = st.selectbox(
        "Pilih kolom target Anda (y):",
        options=df.columns.tolist(),
        index=0
    )
    st.warning(f"Menggunakan '{target_column_name}' sebagai kolom target berdasarkan pilihan pengguna.")
else:
    st.info(f"Menggunakan '{target_column_name}' sebagai kolom target yang ditemukan di data.")

# --- Initial Data Insights: Perbandingan CO2 Emissions(g/km) dengan Merek Umum (tetap sama) ---
if 'Make' in df.columns and target_column_name in df.columns:
    st.subheader(f"Perbandingan {target_column_name} Rata-rata per Merek Umum")
    num_common_makes = st.slider(
        "Pilih jumlah merek mobil paling umum yang akan ditampilkan:",
        min_value=5, max_value=min(25, df['Make'].nunique()), value=10, step=1, key='make_slider'
    )
    common_makes = df['Make'].value_counts().nlargest(num_common_makes).index.tolist()
    df_common_makes_filtered = df[df['Make'].isin(common_makes)]
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

st.subheader("Detail Preprocessing Data")

# --- Modifikasi Bagian Preprocessing untuk Melatih & Menyimpan Preprocessor ---
# Ini akan dijalankan sekali saat aplikasi pertama kali dimuat/di-deploy

columns_to_drop_from_notebook = [
    'Model', 'Make', 'Fuel Consumption (City (L/100 km))',
    'Fuel Consumption(Hwy (L/100 km))', 'Fuel Consumption(Comb (L/100 km))'
]
columns_to_drop_existing = [col for col in columns_to_drop_from_notebook if col in df.columns]
if target_column_name in columns_to_drop_existing:
    columns_to_drop_existing.remove(target_column_name)
    st.warning(f"Menghapus kolom target '{target_column_name}' dari daftar drop awal untuk mencegah penghapusan yang tidak disengaja.")
df_processed = df.drop(columns=columns_to_drop_existing, errors='ignore')
st.write(f"Kolom yang dihapus berdasarkan notebook: {columns_to_drop_existing}")
st.write("Kolom setelah penghapusan awal:", df_processed.columns.tolist())

categorical_cols_for_le = ['Fuel Type', 'Transmission', 'Vehicle Class']
categorical_cols_to_encode = [
    col for col in categorical_cols_for_le
    if col in df_processed.columns and df_processed[col].dtype == 'object'
]

# Use st.cache_resource for preprocessors so they are loaded once
@st.cache_resource
def get_fitted_preprocessors(df_proc, cat_cols_to_encode, target_col):
    fitted_les = {}
    temp_df_proc = df_proc.copy() # Work on a copy to avoid modifying original df_processed passed to cache

    for col in cat_cols_to_encode:
        le = LabelEncoder()
        if temp_df_proc[col].isnull().any():
            temp_df_proc[col] = temp_df_proc[col].fillna('Missing')
        temp_df_proc[col] = le.fit_transform(temp_df_proc[col])
        fitted_les[col] = le
    
    # Separate X and y for scaler fitting
    X_for_scaler = temp_df_proc.drop(columns=[target_col])
    
    num_cols_for_scaling = X_for_scaler.select_dtypes(include=np.number).columns
    scaler_obj = None
    if len(num_cols_for_scaling) > 0:
        scaler_obj = StandardScaler()
        scaler_obj.fit(X_for_scaler[num_cols_for_scaling]) # Fit scaler on X before full split
    
    return fitted_les, scaler_obj

fitted_label_encoders, scaler = get_fitted_preprocessors(df_processed, categorical_cols_to_encode, target_column_name)

# Apply encoding and scaling to df_processed now
if fitted_label_encoders:
    st.write(f"Menerapkan Label Encoding pada: {categorical_cols_to_encode}")
    for col in categorical_cols_to_encode:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna('Missing')
        df_processed[col] = fitted_label_encoders[col].transform(df_processed[col])
    st.write("Label Encoding berhasil diterapkan.")

if scaler:
    st.write("Fitur numerik telah diskalakan menggunakan StandardScaler.")
    numerical_cols_to_scale = df_processed.select_dtypes(include=np.number).columns.tolist()
    # Remove target column if it's in the numerical columns to be scaled
    if target_column_name in numerical_cols_to_scale:
        numerical_cols_to_scale.remove(target_column_name)
    if numerical_cols_to_scale:
        df_processed[numerical_cols_to_scale] = scaler.transform(df_processed[numerical_cols_to_scale])
    else:
        st.info("Tidak ada kolom numerik untuk diskalakan setelah Label Encoding.")

st.write("Kolom setelah Label Encoding dan Scaling:", df_processed.columns.tolist())

# --- Validasi Kolom Target (tetap sama) ---
if not pd.api.types.is_numeric_dtype(df_processed[target_column_name]):
    st.error(f"Kolom target '{target_column_name}' mengandung nilai non-numerik. Pastikan itu numerik untuk regresi.")
    st.stop()

# --- Pemisahan X dan y (tetap sama) ---
X = df_processed.drop(columns=[target_column_name])
y = df_processed[target_column_name]

X.replace([np.inf, -np.inf], np.nan, inplace=True)
initial_rows_X = X.shape[0]
X.dropna(inplace=True)
y = y.loc[X.index]
if X.shape[0] < initial_rows_X:
    st.warning(f"Menghapus {initial_rows_X - X.shape[0]} baris yang mengandung nilai NaN/Inf di fitur setelah preprocessing.")

st.write("Bentuk X (fitur) setelah pemisahan dan penanganan NaN:", X.shape)
st.write("Bentuk y (target) setelah pemisahan dan penanganan NaN:", y.shape)
st.dataframe(X.head())

# --- Distribusi Emisi Karbon (tetap sama) ---
st.subheader(f"Distribusi '{target_column_name}'")
fig_dist_co2, ax_dist_co2 = plt.subplots(figsize=(10, 6))
sns.histplot(y, kde=True, ax=ax_dist_co2, bins=30)
ax_dist_co2.set_title(f"Distribusi '{target_column_name}'")
ax_dist_co2.set_xlabel(target_column_name)
ax_dist_co2.set_ylabel("Frekuensi")
st.pyplot(fig_dist_co2)
st.markdown("---")

# --- Heatmap Korelasi (tetap sama) ---
st.subheader("Heatmap Korelasi Antar Fitur dan Target")
df_for_corr = pd.concat([X, y], axis=1)
correlation_matrix = df_for_corr.corr(numeric_only=True)
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
ax_corr.set_title("Heatmap Korelasi Antar Fitur dan Target")
st.pyplot(fig_corr)
st.markdown("---")

# --- Train-Test Split (tetap sama) ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_cols_order = X_train.columns.tolist() 

st.success(f"Data dibagi menjadi X_train ({X_train.shape}), X_test ({X_test.shape}), y_train ({y_train.shape}), y_test ({y_test.shape})")
st.write("X_train.head():")
st.dataframe(X_train.head())
st.write("y_train.head():")
st.dataframe(y_train.head())


st.markdown("---")

# --- 3. Model Training & Evaluation (Diubah untuk opsional train/load) ---
st.header("3. Pelatihan & Evaluasi Model (Latih atau Muat Model Tersimpan)")

model_choice = st.selectbox(
    "Pilih Model Regresi:",
    ('XGBoost', 'Lasso Regression', 'Linear Regression', 'Support Vector Regressor (SVR)'),
    key='train_model_select'
)

# Gunakan st.cache_resource untuk model yang dimuat
@st.cache_resource
def get_trained_model(model_name, X_train_data, y_train_data):
    model = None
    try:
        # Coba muat model yang sudah ada
        model_path = f"{MODEL_PATH_PREFIX}{model_name}.pkl"
        if os.path.exists(model_path):
            st.info(f"Memuat model '{model_name}' yang sudah dilatih...")
            model = joblib.load(model_path)
            st.success("Model berhasil dimuat.")
        else:
            st.warning(f"Model '{model_name}' tidak ditemukan di {model_path}. Melatih model baru.")
            raise FileNotFoundError # Paksa pelatihan jika tidak ditemukan
    except (FileNotFoundError, Exception) as e:
        # Jika gagal memuat, latih model baru
        with st.spinner(f"Melatih model {model_name} baru..."):
            if model_name == 'XGBoost':
                model = xgb.XGBRegressor(objective='reg:squareerror', n_estimators=100, random_state=42)
            elif model_name == 'Lasso Regression':
                model = Lasso(alpha=0.1, random_state=42)
            elif model_name == 'Linear Regression':
                model = LinearRegression()
            elif model_name == 'Support Vector Regressor (SVR)':
                model = SVR(kernel='rbf', C=100)
            model.fit(X_train_data, y_train_data)
            joblib.dump(model, model_path)
            st.success(f"Model '{model_name}' baru dilatih dan disimpan di {model_path}")
    return model

# Latih/muat model saat aplikasi dimuat
model = get_trained_model(model_choice, X_train, y_train)

if model is not None:
    y_pred = model.predict(X_test)
    metrics = {
        'mse': mean_squared_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    st.subheader(f"Metrik Evaluasi untuk {model_choice}:")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Mean Squared Error (MSE)", f"{metrics['mse']:.4f}")
    with col2: st.metric("R-Squared (R2)", f"{metrics['r2']:.4f}")
    with col3: st.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}")
    with col4: st.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.4f}")
    st.markdown("---")
else:
    st.error("Gagal memuat atau melatih model.")
    st.stop()


# --- 4. Visualizations (tetap sama) ---
st.header("4. Visualisasi Hasil Model")

st.subheader(f"Plot antara Fitur dan {target_column_name}")
feature_to_plot = st.selectbox(
    "Pilih Fitur untuk di-plot terhadap target:",
    options=X.columns.tolist(),
    key='feature_target_plot_select'
)
if feature_to_plot:
    fig_feature_target, ax_feature_target = plt.subplots(figsize=(10, 6))
    if pd.api.types.is_numeric_dtype(X[feature_to_plot]):
        sns.scatterplot(x=X[feature_to_plot], y=y, ax=ax_feature_target, alpha=0.6)
        ax_feature_target.set_title(f"Scatter Plot: {feature_to_plot} vs {target_column_name}")
    else:
        sns.boxplot(x=X[feature_to_plot], y=y, ax=ax_feature_target)
        ax_feature_target.set_title(f"Box Plot: {feature_to_plot} vs {target_column_name}")
    ax_feature_target.set_xlabel(feature_to_plot)
    ax_feature_target.set_ylabel(target_column_name)
    st.pyplot(fig_feature_target)
st.markdown("---")


st.subheader("Aktual vs. Prediksi")
fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
ax_scatter.scatter(y_test, y_pred, alpha=0.7)
ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax_scatter.set_xlabel(f"Nilai Aktual {target_column_name}")
ax_scatter.set_ylabel(f"Nilai Prediksi {target_column_name}")
ax_scatter.set_title(f"Nilai Aktual vs. Prediksi (Test Set) - {model_choice}")
st.pyplot(fig_scatter)

st.subheader("Plot Residual")
residuals = y_test - y_pred
fig_residuals, ax_residuals = plt.subplots(figsize=(10, 6))
ax_residuals.scatter(y_pred, residuals, alpha=0.7)
ax_residuals.axhline(y=0, color='r', linestyle='--', lw=2)
ax_residuals.set_xlabel(f"Nilai Prediksi {target_column_name}")
ax_residuals.set_ylabel("Residual (Aktual - Prediksi)")
ax_residuals.set_title(f"Plot Residual - {model_choice}")
st.pyplot(fig_residuals)

st.subheader("Distribusi Residual")
fig_hist_residuals, ax_hist_residuals = plt.subplots(figsize=(10, 6))
sns.histplot(residuals, kde=True, ax=ax_hist_residuals)
ax_hist_residuals.set_xlabel("Residual")
ax_hist_residuals.set_ylabel("Frekuensi")
ax_hist_residuals.set_title(f"Distribusi Residual - {model_choice}")
st.pyplot(fig_hist_residuals)

st.subheader("QQ-Plot Residual")
fig_qq, ax_qq = plt.subplots(figsize=(8, 8))
sm.qqplot(residuals, line='s', ax=ax_qq)
ax_qq.set_title(f"QQ-Plot Residual - {model_choice}")
st.pyplot(fig_qq)
st.markdown("---")

if model_choice == 'XGBoost':
    st.subheader("Pentingnya Fitur (XGBoost)")
    feature_importances = model.feature_importances_
    feature_names = X.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    fig_feature_imp, ax_feature_imp = plt.subplots(figsize=(12, 7))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax_feature_imp)
    ax_feature_imp.set_title("15 Fitur Terpenting (XGBoost)")
    ax_feature_imp.set_xlabel("Skor Kepentingan")
    ax_feature_imp.set_ylabel("Fitur")
    st.pyplot(fig_feature_imp)
else:
    st.info("Pentingnya fitur biasanya ditampilkan untuk model berbasis pohon seperti XGBoost.")

st.subheader("Contoh Nilai Aktual vs. Prediksi")
results_df = pd.DataFrame({
    'Aktual': y_test,
    'Prediksi': y_pred,
    'Residual': residuals
}).head(15)
st.dataframe(results_df)

# --- NEW SECTION: Make a New Prediction ---
st.markdown("---")
st.header("5. Buat Prediksi Baru")

# --- Muat Preprocessor dan Model yang Tersimpan ---
# Ini akan dimuat sekali saat aplikasi dimulai, berkat st.cache_resource

# @st.cache_resource
# def load_cached_preprocessors():
#     try:
#         loaded_scaler_obj = joblib.load(SCALER_PATH)
#         loaded_les_dict = joblib.load(LABEL_ENCODERS_PATH)
#         return loaded_scaler_obj, loaded_les_dict
#     except FileNotFoundError:
#         st.warning("File preprocessor tidak ditemukan. Harap latih model di bagian 'Pelatihan & Evaluasi Model' terlebih dahulu.")
#         return None, None
#     except Exception as e:
#         st.warning(f"Error memuat preprocessor: {e}. Harap latih model terlebih dahulu.")
#         return None, None

# loaded_scaler, loaded_label_encoders = load_cached_preprocessors()

# Perhatikan: di sini kita menggunakan `scaler` dan `fitted_label_encoders`
# yang sudah diinisialisasi dan di-fit dari bagian preprocessing di atas.
# Karena mereka sudah di-cache dengan `@st.cache_resource` atau ada di scope global
# setelah di-fit (jika tidak menggunakan cache di sana), kita bisa langsung menggunakannya.
# Jika ada masalah deployment karena scope, kita harus memuatnya dari file.
# Untuk deployment yang robust, memuat dari file yang sudah ada adalah praktik terbaik.

# Mari kita tetap muat dari file agar lebih eksplisit untuk deployment
# Fungsi loading ini juga akan di-cache

@st.cache_resource
def load_trained_model_for_prediction(model_name):
    model_path = f"{MODEL_PATH_PREFIX}{model_name}.pkl"
    try:
        if os.path.exists(model_path):
            st.info(f"Memuat model '{model_name}' untuk prediksi...")
            model = joblib.load(model_path)
            st.success("Model berhasil dimuat.")
            return model
        else:
            st.error(f"Model '{model_name}' tidak ditemukan di {model_path}. Harap latih model di atas.")
            return None
    except Exception as e:
        st.error(f"Gagal memuat model '{model_name}': {e}. Pastikan model sudah dilatih dan disimpan.")
        return None

# Tampilkan opsi model yang tersedia untuk prediksi
available_models = []
try:
    for f in os.listdir('.'): 
        if f.startswith(MODEL_PATH_PREFIX) and f.endswith('.pkl'):
            available_models.append(f.replace(MODEL_PATH_PREFIX, '').replace('.pkl', ''))
except Exception as e:
    st.error(f"Gagal mencari model tersimpan untuk prediksi: {e}")

if not available_models:
    st.warning("Tidak ada model yang tersimpan untuk prediksi. Harap latih model di bagian 'Pelatihan & Evaluasi Model' terlebih dahulu.")

if available_models and scaler and fitted_label_encoders: # Check if preprocessors are ready
    model_to_predict_with = st.selectbox(
        "Pilih Model untuk Prediksi:",
        options=available_models,
        key='predict_model_select' # Kunci unik
    )
    loaded_model = load_trained_model_for_prediction(model_to_predict_with)

    if loaded_model:
        st.write("Masukkan nilai fitur untuk mendapatkan prediksi emisi CO2:")

        col_input1, col_input2, col_input3 = st.columns(3)

        make_options = df['Make'].unique().tolist() if 'Make' in df.columns else []
        transmission_options = df['Transmission'].unique().tolist() if 'Transmission' in df.columns else []
        fuel_type_options = df['Fuel Type'].unique().tolist() if 'Fuel Type' in df.columns else []
        vehicle_class_options = df['Vehicle Class'].unique().tolist() if 'Vehicle Class' in df.columns else []

        with col_input1:
            input_make = st.selectbox('Merk Mobil', options=make_options, key='input_make_pred')
            
            # Dynamic Model selection based on Make
            filtered_models = df[df['Make'] == input_make]['Model'].unique().tolist() if input_make else []
            input_model = st.selectbox('Tipe Mobil', options=filtered_models, key='input_model_pred')
            input_engine_size = st.number_input('Ukuran Mesin (L)', min_value=0.5, max_value=8.0, value=2.0, step=0.1, key='input_engine_size_pred')

        with col_input2:
            input_cylinders = st.number_input('Jumlah Silinder', min_value=3, max_value=12, value=4, step=1, key='input_cylinders_pred')
            input_transmission = st.selectbox('Transmisi', options=transmission_options, key='input_transmission_pred')
            input_fuel_type = st.selectbox('Tipe Bahan Bakar', options=fuel_type_options, key='input_fuel_type_pred')

        with col_input3:
            input_vehicle_class = st.selectbox('Kelas Kendaraan', options=vehicle_class_options, key='input_vehicle_class_pred')
            # 'Fuel Consumption(Comb (L/100km))' is dropped based on notebook
            # if you want to include it, ensure it's removed from 'columns_to_drop_from_notebook'
            # input_fuel_comb = st.number_input('Konsumsi Bahan Bakar (L/100km)', min_value=5.0, max_value=25.0, value=10.0, step=0.1, key='input_fuel_comb_pred')

        if st.button("Dapatkan Prediksi"):
            try:
                # Create raw input dictionary
                raw_input_data = {
                    'Engine Size(L)': input_engine_size,
                    'Cylinders': input_cylinders,
                    'Transmission': input_transmission,
                    'Fuel Type': input_fuel_type,
                    'Vehicle Class': input_vehicle_class,
                    # If 'Fuel Consumption(Comb (L/100km))' is an actual feature in X_train_cols_order:
                    # 'Fuel Consumption(Comb (L/100km))': input_fuel_comb,
                }
                input_df_raw = pd.DataFrame([raw_input_data])
                
                # Apply Label Encoding using the loaded encoders
                for col_name, encoder in fitted_label_encoders.items():
                    if col_name in input_df_raw.columns:
                        if input_df_raw[col_name].iloc[0] not in encoder.classes_:
                            st.error(f"Kategori '{input_df_raw[col_name].iloc[0]}' di kolom '{col_name}' tidak dikenali oleh model. Harap pilih dari daftar yang tersedia dari data pelatihan.")
                            st.stop()
                        input_df_raw[col_name] = encoder.transform(input_df_raw[col_name])
                
                # Ensure column order matches X_train_cols_order and fill missing with 0
                final_input_df = pd.DataFrame(columns=X_train_cols_order)
                for col in X_train_cols_order:
                    if col in input_df_raw.columns:
                        final_input_df[col] = input_df_raw[col]
                    else:
                        # Default to 0 for features not provided by user but expected by model
                        # This could also be a mean/median from X_train
                        final_input_df[col] = 0 

                # Apply StandardScaler using the loaded scaler
                numerical_cols_to_scale_for_pred = final_input_df.select_dtypes(include=np.number).columns
                if scaler and len(numerical_cols_to_scale_for_pred) > 0:
                    final_input_df[numerical_cols_to_scale_for_pred] = scaler.transform(final_input_df[numerical_cols_to_scale_for_pred])
                
                # Make prediction
                prediction = loaded_model.predict(final_input_df)
                
                st.success("Prediksi Emisi CO2:")
                st.write(f"**{prediction[0]:.2f} g/km**")
                
            except Exception as e:
                st.error(f"Gagal membuat prediksi: {e}.")
                st.error("Pastikan semua input valid dan preprocessing benar.")
                st.write("Debug Info:")
                st.write(f"Input DataFrame (before final processing): {input_df_raw}")
                if 'final_input_df' in locals():
                    st.write(f"Final Input DataFrame columns (for model): {final_input_df.columns.tolist()}")
                    st.write(f"Final Input DataFrame head: {final_input_df.head()}")
                st.write(f"Expected X_train columns: {X_train_cols_order}")
