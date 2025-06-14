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
            st.warning("Tidak ada file model .pkl yang ditemukan di repositori.")
        return loaded_scaler, loaded_label_encoders, available_models_list
    except FileNotFoundError:
        st.error("File preprocessor (.pkl) tidak ditemukan. Harap pastikan Anda telah mengunggahnya.")
        st.stop()
    except Exception as e:
        st.error(f"Error memuat sumber daya: {e}.")
        st.stop()

# --- Load Resources ---
scaler, encoders, available_models = load_preprocessors_and_available_models(SCALER_PATH, LABEL_ENCODERS_PATH, MODEL_PATH_PREFIX)
X_train_cols_order = [] 

# --- 1. Data Loading & Exploratory Data Analysis (EDA) ---
st.header("1. Analisis Data Eksploratif")

df = load_data(DATA_FILE_PATH)
df_original = df.copy()
target_column_name = 'CO2 Emissions(g/km)' 

### BAGIAN DATA LOADING DITAMBAHKAN KEMBALI ###
st.subheader("Tampilan Awal Data")
st.info(f"Data dari '{DATA_FILE_PATH}' berhasil dimuat. Menampilkan 5 baris pertama:")
st.dataframe(df.head())
st.markdown("---")

### PLOT DISTRIBUSI CO2 DITAMBAHKAN KEMBALI ###
st.subheader(f"Distribusi Emisi CO2")
fig_dist_co2, ax_dist_co2 = plt.subplots(figsize=(10, 6))
sns.histplot(df_original[target_column_name], kde=True, ax=ax_dist_co2, bins=30)
ax_dist_co2.set_title(f"Distribusi '{target_column_name}'")
ax_dist_co2.set_xlabel(target_column_name)
ax_dist_co2.set_ylabel("Frekuensi")
st.pyplot(fig_dist_co2)
st.markdown("---")


if 'Make' in df.columns and target_column_name in df.columns:
    st.subheader(f"Perbandingan Emisi CO2 Rata-rata per Merek")
    num_common_makes = st.slider(
        "Pilih jumlah merek mobil untuk ditampilkan:",
        min_value=5, max_value=min(25, df_original['Make'].nunique()), value=10, key='make_slider'
    )
    common_makes = df_original['Make'].value_counts().nlargest(num_common_makes).index
    df_common_makes = df_original[df_original['Make'].isin(common_makes)]
    make_avg_co2 = df_common_makes.groupby('Make')[target_column_name].mean().sort_values(ascending=False)
    
    fig_make_co2, ax_make_co2 = plt.subplots(figsize=(14, 7))
    sns.barplot(x=make_avg_co2.index, y=make_avg_co2.values, ax=ax_make_co2)
    ax_make_co2.set_title(f"Rata-rata Emisi CO2 per {num_common_makes} Merek Mobil Teratas")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_make_co2)
    st.markdown("---")

st.subheader("Heatmap Korelasi Antar Fitur Numerik")
numeric_df = df_original.select_dtypes(include=np.number)
correlation_matrix = numeric_df.corr()
fig_corr, ax_corr = plt.subplots(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax_corr)
ax_corr.set_title("Heatmap Korelasi Antar Fitur Numerik (Data Asli)")
st.pyplot(fig_corr)
st.markdown("---")


# --- 2. Preprocessing & Splitting (Proses Latar Belakang) ---
df_processed = df.copy()
columns_to_drop = ['Model', 'Make']
df_processed = df_processed.drop(columns=[col for col in columns_to_drop if col in df_processed.columns], errors='ignore')
X_temp = df_processed.drop(columns=[target_column_name])
y_temp = df_processed[target_column_name]
numerical_cols_temp = X_temp.select_dtypes(include=np.number).columns
if scaler and len(numerical_cols_temp) > 0:
    X_temp[numerical_cols_temp] = scaler.transform(X_temp[numerical_cols_temp])
if not numerical_cols_temp.empty:
    for col in numerical_cols_temp:
        Q1, Q3 = X_temp[col].quantile(0.25), X_temp[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        filter_mask = (X_temp[col] >= lower_bound) & (X_temp[col] <= upper_bound)
        X_temp = X_temp[filter_mask]
        y_temp = y_temp.loc[X_temp.index]
categorical_cols_for_le = ['Fuel Type', 'Transmission', 'Vehicle Class']
available_cat_cols_for_le = [col for col in categorical_cols_for_le if col in X_temp.columns and X_temp[col].dtype == 'object']
if encoders and available_cat_cols_for_le:
    for col in available_cat_cols_for_le:
        X_temp[col] = encoders[col].transform(X_temp[col].fillna('Missing'))
X_temp.replace([np.inf, -np.inf], np.nan, inplace=True)
y_temp = y_temp.loc[X_temp.dropna().index]
X_final = X_temp.loc[y_temp.index]
y_final = y_temp
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
X_train_cols_order = X_train.columns.tolist()

# --- 3. Model Evaluation ---
st.header("3. Evaluasi Model")

if available_models:
    model_choice_eval = st.selectbox("Pilih Model untuk Evaluasi:", options=available_models, key='eval_model_select')
    model_for_evaluation = load_trained_model(model_choice_eval, MODEL_PATH_PREFIX)
    
    if model_for_evaluation:
        y_pred_eval = model_for_evaluation.predict(X_test)
        metrics = {
            'r2': r2_score(y_test, y_pred_eval),
            'mae': mean_absolute_error(y_test, y_pred_eval),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_eval))
        }

        st.subheader(f"Metrik Evaluasi untuk {model_choice_eval}:")
        col1, col2, col3 = st.columns(3)
        col1.metric("R-Squared (R²)", f"{metrics['r2']:.4f}")
        col2.metric("Mean Absolute Error (MAE)", f"{metrics['mae']:.4f}")
        col3.metric("Root Mean Squared Error (RMSE)", f"{metrics['rmse']:.4f}")
        st.markdown("---")

        # --- 4. Visualisasi Hasil Model ---
        st.header("4. Visualisasi Hasil Model")
        
        viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Aktual vs. Prediksi", "🔍 Analisis Residual", "💡 Pentingnya Fitur"])

        with viz_tab1:
            st.subheader("Plot Aktual vs. Prediksi")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred_eval, alpha=0.7)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel(f"Nilai Aktual {target_column_name}")
            ax.set_ylabel(f"Nilai Prediksi {target_column_name}")
            ax.set_title(f"Aktual vs. Prediksi - {model_choice_eval}")
            st.pyplot(fig)

        with viz_tab2:
            st.subheader("Plot Residual")
            residuals = y_test - y_pred_eval
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_pred_eval, residuals, alpha=0.7)
            ax.axhline(y=0, color='r', linestyle='--', lw=2)
            ax.set_xlabel(f"Nilai Prediksi {target_column_name}")
            ax.set_ylabel("Residual (Aktual - Prediksi)")
            ax.set_title(f"Plot Residual - {model_choice_eval}")
            st.pyplot(fig)
            
            st.subheader("QQ-Plot dari Residual")
            fig_qq, ax_qq = plt.subplots(figsize=(6, 6))
            sm.qqplot(residuals, line='s', ax=ax_qq)
            ax_qq.set_title(f"QQ-Plot Residual - {model_choice_eval}")
            st.pyplot(fig_qq)

        with viz_tab3:
            st.subheader("Pentingnya Fitur Model")
            if model_choice_eval == 'XGBoost' and hasattr(model_for_evaluation, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': X_train_cols_order,
                    'Importance': model_for_evaluation.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 7))
                sns.barplot(x='Importance', y='Feature', data=importance_df.head(15), ax=ax)
                ax.set_title("15 Fitur Terpenting (XGBoost)")
                st.pyplot(fig)
            else:
                st.info("Pentingnya fitur hanya dapat ditampilkan untuk model berbasis pohon seperti XGBoost.")
else:
    st.info("Tidak ada model yang tersedia untuk evaluasi. Harap pastikan file .pkl sudah terunggah.")
st.markdown("---")

# --- 5. Buat Prediksi Baru (Interaktif) ---
st.header("5. Buat Prediksi Baru")

if scaler and encoders and available_models and X_train_cols_order:
    model_to_predict_with = st.selectbox("Pilih Model untuk Prediksi Interaktif:", options=available_models, key='predict_model_select')
    loaded_model_predict = load_trained_model(model_to_predict_with, MODEL_PATH_PREFIX)

    if loaded_model_predict:
        st.write("Pilih kombinasi kendaraan untuk mendapatkan prediksi emisi CO2:")
        col1, col2 = st.columns(2)
        makes = sorted(df_original['Make'].unique())
        selected_make = col1.selectbox("Pilih Merek Mobil:", options=makes, key='pred_make')
        models = sorted(df_original[df_original['Make'] == selected_make]['Model'].unique())
        selected_model = col2.selectbox("Pilih Model Mobil:", options=models, key='pred_model')
        df_filtered_options = df_original[(df_original['Make'] == selected_make) & (df_original['Model'] == selected_model)]
        col3, col4 = st.columns(2)
        transmissions = sorted(df_filtered_options['Transmission'].unique())
        selected_transmission = col3.selectbox("Pilih Transmisi:", options=transmissions, key='pred_transmission')
        fuel_types = sorted(df_filtered_options[df_filtered_options['Transmission'] == selected_transmission]['Fuel Type'].unique())
        selected_fuel_type = col4.selectbox("Pilih Tipe Bahan Bakar:", options=fuel_types, key='pred_fuel')

        if st.button("Dapatkan Prediksi"):
            input_row = df_original[
                (df_original['Make'] == selected_make) & (df_original['Model'] == selected_model) &
                (df_original['Transmission'] == selected_transmission) & (df_original['Fuel Type'] == selected_fuel_type)
            ]
            if not input_row.empty:
                input_features_raw = input_row.iloc[[0]]
                actual_emission = input_features_raw[target_column_name].values[0]
                input_df_processed = input_features_raw.copy()
                try:
                    for col_name, encoder_obj in encoders.items():
                        if col_name in input_df_processed.columns:
                            input_df_processed[col_name] = encoder_obj.transform(input_df_processed[col_name])
                    numerical_cols_for_input_scaling = [col for col in numerical_cols_temp if col in input_df_processed.columns]
                    if scaler and numerical_cols_for_input_scaling:
                        input_df_processed[numerical_cols_for_input_scaling] = scaler.transform(input_df_processed[numerical_cols_for_input_scaling])
                    final_input_df = input_df_processed[X_train_cols_order]
                    prediction = loaded_model_predict.predict(final_input_df)
                    st.success("Prediksi Berhasil!")
                    res_col1, res_col2 = st.columns(2)
                    res_col1.metric("Nilai Emisi Aktual (dari data)", f"{actual_emission:.2f} g/km")
                    res_col2.metric("Nilai Emisi Prediksi", f"{prediction[0]:.2f} g/km", delta=f"{prediction[0] - actual_emission:.2f}")
                except Exception as e:
                    st.error(f"Gagal membuat prediksi: {e}")
            else:
                st.error("Kombinasi kendaraan tidak ditemukan.")
else:
    st.info("Sumber daya tidak siap untuk prediksi. Pastikan file .pkl sudah terunggah.")
