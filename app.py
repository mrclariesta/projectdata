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

st.set_page_config(layout="wide")

# --- Initialize Session State ---
# This is crucial for storing all results and preventing them from disappearing.
def initialize_session_state():
    keys_to_init = {
        'model': None,
        'scaler': None,
        'encoders': {},
        'X_columns': None,
        'evaluation_run': False,
        'metrics': {},
        'y_test': None,
        'y_pred': None,
        'residuals': None,
        'model_choice': 'XGBoost',
        'interactive_prediction': None
    }
    for key, default_value in keys_to_init.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

st.title("Vehicle Emission Prediction Dashboard")
st.markdown("---")

# --- 1. Load Data ---
st.header("1. Data Loading & Preprocessing")

data_file_path = 'emission.csv'

@st.cache_data
def load_data(path):
    try:
        data = pd.read_csv(path)
        return data
    except FileNotFoundError:
        st.error(f"File data tidak ditemukan di: {path}.")
        st.stop()
    except Exception as e:
        st.error(f"Terjadi kesalahan saat memuat file: {e}")
        st.stop()

df = load_data(data_file_path)
df_original = df.copy()

if df is not None:
    st.success(f"File '{data_file_path}' berhasil dimuat!")
    st.dataframe(df.head())

    target_column_name = 'CO2 Emissions(g/km)'
    if target_column_name not in df.columns:
        st.error(f"Kolom target default '{target_column_name}' tidak ditemukan.")
        st.stop()

    # --- 2. Exploratory Data Analysis (EDA) ---
    st.header("2. Analisis Data Eksploratif")

    # --- Comparison by Make ---
    st.subheader(f"Perbandingan {target_column_name} Rata-rata per Merek")
    num_common_makes = st.slider(
        "Pilih jumlah merek mobil untuk ditampilkan:",
        min_value=5, max_value=min(25, df['Make'].nunique()), value=10, key='make_slider'
    )
    common_makes = df['Make'].value_counts().nlargest(num_common_makes).index
    df_common_makes = df[df['Make'].isin(common_makes)]
    make_avg_co2 = df_common_makes.groupby('Make')[target_column_name].mean().sort_values(ascending=False)
    
    fig_make_co2, ax_make_co2 = plt.subplots(figsize=(14, 7))
    sns.barplot(x=make_avg_co2.index, y=make_avg_co2.values, ax=ax_make_co2)
    ax_make_co2.set_title(f"Rata-rata {target_column_name} per {num_common_makes} Merek Mobil Teratas")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_make_co2)
    st.markdown("---")

    # --- Distribution by Categorical Features ---
    st.subheader(f"Distribusi '{target_column_name}' per Fitur Kategorikal")
    cat_features = ['Fuel Type', 'Transmission', 'Vehicle Class']
    available_cat_features = [col for col in cat_features if col in df.columns]

    selected_cat_feature = st.selectbox("Pilih Fitur Kategorikal:", options=available_cat_features)
    plot_type = st.radio("Pilih Jenis Plot:", ('Box Plot', 'Violin Plot'), key='cat_plot_type')
    
    fig_cat_target, ax_cat_target = plt.subplots(figsize=(12, 7))
    if plot_type == 'Box Plot':
        sns.boxplot(x=selected_cat_feature, y=target_column_name, data=df, ax=ax_cat_target)
    else:
        sns.violinplot(x=selected_cat_feature, y=target_column_name, data=df, ax=ax_cat_target)
    ax_cat_target.set_title(f"{plot_type}: {target_column_name} per {selected_cat_feature}")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig_cat_target)
    st.markdown("---")
    
    # Preprocessing is done here but not displayed in detail to keep the UI clean
    # The actual processing happens before model training.
    columns_to_drop = ['Model', 'Make', 'Fuel Consumption (City (L/100 km))', 'Fuel Consumption(Hwy (L/100 km))', 'Fuel Consumption(Comb (L/100 km))']
    df_processed = df.drop(columns=[col for col in columns_to_drop if col in df.columns], errors='ignore')

    encoders = {}
    for col in available_cat_features:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col].fillna('Missing'))
        encoders[col] = le
    st.session_state.encoders = encoders
    
    X = df_processed.drop(columns=[target_column_name])
    y = df_processed[target_column_name]
    X.dropna(inplace=True)
    y = y.loc[X.index]
    st.session_state.X_columns = X.columns.tolist()

    scaler = StandardScaler()
    numerical_cols = X.select_dtypes(include=np.number).columns
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    st.session_state.scaler = scaler

    X_train, X_test, y_train, y_test_series = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Model Training ---
st.header("3. Pelatihan Model")

model_choice = st.selectbox(
    "Pilih Model Regresi:",
    ('XGBoost', 'Lasso Regression', 'Linear Regression', 'Support Vector Regressor (SVR)'),
    key='model_selector'
)
st.session_state.model_choice = model_choice # Store the choice

if st.button("Latih Model", type="primary"):
    with st.spinner(f"Melatih model {st.session_state.model_choice}..."):
        if st.session_state.model_choice == 'XGBoost':
            model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
        elif st.session_state.model_choice == 'Lasso Regression':
            model = Lasso(alpha=0.1, random_state=42)
        elif st.session_state.model_choice == 'Linear Regression':
            model = LinearRegression()
        else: # SVR
            model = SVR(kernel='rbf', C=100)
        
        model.fit(X_train, y_train)
        
        # Store everything in session state
        st.session_state.model = model
        y_pred = model.predict(X_test)
        st.session_state.y_test = y_test_series
        st.session_state.y_pred = y_pred
        st.session_state.residuals = y_test_series - y_pred
        
        st.session_state.metrics = {
            'mse': mean_squared_error(y_test_series, y_pred),
            'r2': r2_score(y_test_series, y_pred),
            'mae': mean_absolute_error(y_test_series, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_series, y_pred))
        }
        st.session_state.evaluation_run = True # Flag that evaluation is done
    st.success(f"Model {st.session_state.model_choice} berhasil dilatih dan dievaluasi!")

st.markdown("---")

# --- 4. Evaluation and Visualization ---
st.header("4. Evaluasi & Visualisasi Hasil")
if not st.session_state.evaluation_run:
    st.info("Silakan latih model di atas untuk melihat hasil evaluasi dan visualisasi.")
else:
    st.subheader(f"Metrik Evaluasi untuk {st.session_state.model_choice}:")
    m = st.session_state.metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R-Squared (R2)", f"{m['r2']:.4f}")
    col2.metric("Mean Absolute Error (MAE)", f"{m['mae']:.4f}")
    col3.metric("Mean Squared Error (MSE)", f"{m['mse']:.4f}")
    col4.metric("Root Mean Squared Error (RMSE)", f"{m['rmse']:.4f}")

    st.subheader("Visualisasi Hasil")
    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["Actual vs. Predicted", "Residual Plot", "Residual Distribution", "Feature Importance"])

    with viz_tab1:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(st.session_state.y_test, st.session_state.y_pred, alpha=0.7)
        ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()], [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--', lw=2)
        ax.set_xlabel(f"Nilai Aktual {target_column_name}")
        ax.set_ylabel(f"Nilai Prediksi {target_column_name}")
        ax.set_title(f"Aktual vs. Prediksi - {st.session_state.model_choice}")
        st.pyplot(fig)

    with viz_tab2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(st.session_state.y_pred, st.session_state.residuals, alpha=0.7)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        ax.set_xlabel(f"Nilai Prediksi {target_column_name}")
        ax.set_ylabel("Residual (Aktual - Prediksi)")
        ax.set_title(f"Plot Residual - {st.session_state.model_choice}")
        st.pyplot(fig)

    with viz_tab3:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.histplot(st.session_state.residuals, kde=True, ax=ax)
        ax.set_title(f"Distribusi Residual - {st.session_state.model_choice}")
        st.pyplot(fig)

    with viz_tab4:
        if st.session_state.model_choice == 'XGBoost' and hasattr(st.session_state.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'Feature': st.session_state.X_columns,
                'Importance': st.session_state.model.feature_importances_
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
            ax.set_title("Pentingnya Fitur (XGBoost)")
            st.pyplot(fig)
        else:
            st.info("Pentingnya fitur hanya ditampilkan untuk model berbasis pohon seperti XGBoost.")
st.markdown("---")

# --- 5. Interactive Prediction Section ---
st.header("5. Prediksi Interaktif")
if st.session_state.model is None:
    st.warning("Harap latih model di Bagian #3 untuk mengaktifkan prediksi interaktif.")
else:
    st.success(f"Model **{st.session_state.model_choice}** siap untuk prediksi.")
    
    # Dependent dropdowns for a better user experience
    col1, col2 = st.columns(2)
    makes = sorted(df_original['Make'].unique())
    selected_make = col1.selectbox("Pilih Merek Mobil:", options=makes)

    models = sorted(df_original[df_original['Make'] == selected_make]['Model'].unique())
    selected_model = col2.selectbox("Pilih Model Mobil:", options=models)

    df_filtered = df_original[(df_original['Make'] == selected_make) & (df_original['Model'] == selected_model)]
    
    col3, col4 = st.columns(2)
    transmissions = sorted(df_filtered['Transmission'].unique())
    selected_transmission = col3.selectbox("Pilih Transmisi:", options=transmissions)

    fuel_types = sorted(df_filtered[df_filtered['Transmission'] == selected_transmission]['Fuel Type'].unique())
    selected_fuel_type = col4.selectbox("Pilih Tipe Bahan Bakar:", options=fuel_types)
    
    if st.button("Prediksi Emisi CO2"):
        input_row = df_original[
            (df_original['Make'] == selected_make) &
            (df_original['Model'] == selected_model) &
            (df_original['Transmission'] == selected_transmission) &
            (df_original['Fuel Type'] == selected_fuel_type)
        ]

        if not input_row.empty:
            input_features = input_row.iloc[[0]]
            actual_emission = input_features[target_column_name].values[0]

            input_processed = input_features[st.session_state.X_columns].copy()
            for col, le in st.session_state.encoders.items():
                input_processed[col] = le.transform(input_processed[col])
            
            input_processed[numerical_cols] = st.session_state.scaler.transform(input_processed[numerical_cols])

            prediction = st.session_state.model.predict(input_processed)[0]
            st.session_state.interactive_prediction = {'actual': actual_emission, 'predicted': prediction}
        else:
            st.error("Kombinasi tidak ditemukan.")
            st.session_state.interactive_prediction = None
            
    # Display prediction results if they exist in the state
    if st.session_state.interactive_prediction:
        res = st.session_state.interactive_prediction
        st.subheader("Hasil Prediksi Interaktif")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Nilai Aktual (dari data)", f"{res['actual']:.2f} g/km")
        res_col2.metric("Nilai Prediksi (oleh model)", f"{res['predicted']:.2f} g/km", delta=f"{res['predicted'] - res['actual']:.2f}")