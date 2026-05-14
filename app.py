import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
from fpdf import FPDF
import unicodedata
from streamlit_lottie import st_lottie
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

DATA_PATH = "bank (1).csv"
MODEL_PATH = "model.joblib"
COLOR_YES = "#2980b9"
COLOR_NO = "#e74c3c"
COLOR_NAVY = "#1e3c72"
COLOR_LIGHT = "#f0f2f6"

st.set_page_config(page_title="Banka Musteri Davranisi Analizi", layout="wide")

st.markdown(
        f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] {{
        font-family: "Inter", "Trebuchet MS", "Segoe UI", sans-serif;
        background: {COLOR_LIGHT};
        color: #172236;
        font-weight: 600;
    }}
    .hero {{
        background: linear-gradient(135deg, {COLOR_NAVY} 0%, #244a86 60%, #2a5ea1 100%);
        padding: 24px 28px;
        border-radius: 16px;
        color: #ffffff;
        box-shadow: 0 16px 30px rgba(0,0,0,0.18);
    }}
    .card {{
        background: #ffffff;
        border-radius: 10px;
        padding: 18px 20px;
        box-shadow: 0 10px 24px rgba(44, 62, 80, 0.12);
        border: 1px solid #e6e9f0;
    }}
    .card-title {{
        color: #0f1f3a;
        font-weight: 800;
        margin-bottom: 8px;
    }}
    .card h3, .card h4 {{
        color: #0f1f3a;
        font-weight: 800;
    }}
    .kpi {{
        background: #ffffff;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 10px 24px rgba(44, 62, 80, 0.12);
        border: 1px solid #e6e9f0;
    }}
    .kpi-title {{
        color: #56616f;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }}
    .kpi-value {{
        color: {COLOR_NAVY};
        font-size: 26px;
        font-weight: 800;
    }}
    .result-yes {{
        background: {COLOR_YES};
        color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        font-weight: 700;
        border: 2px solid {COLOR_YES};
    }}
    .result-no {{
        background: {COLOR_NO};
        color: #ffffff;
        padding: 16px;
        border-radius: 12px;
        font-weight: 700;
        border: 2px solid {COLOR_NO};
    }}
    .muted {{
        color: #5f6b7a;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] {{
        background: #101826;
        color: #ffffff;
        border-right: 1px solid #0f1a2b;
    }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span,
        section[data-testid="stSidebar"] div {{
            color: #f0f4ff;
            font-weight: 600;
        }}
        .stTabs [data-baseweb="tab"] {{
            color: #e6ecf5;
            font-weight: 700;
        }}
        .stTabs [aria-selected="true"] {{
            color: #ffffff;
        }}
        .stCaption {{
            color: #5f6b7a;
            font-weight: 600;
        }}
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {{
        color: #ffffff;
    }}
    .stButton > button {{
        background: {COLOR_NAVY};
        color: #ffffff;
        border: 1px solid {COLOR_YES};
        border-radius: 10px;
        transition: all 0.2s ease-in-out;
    }}
    .stButton > button:hover {{
        background: #274f91;
        border-color: {COLOR_NO};
        box-shadow: 0 6px 16px rgba(41, 128, 185, 0.35);
    }}
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {{
        border-radius: 8px !important;
    }}
</style>
""",
        unsafe_allow_html=True,
)


def load_lottie(url):
        try:
                response = requests.get(url, timeout=6)
                if response.status_code == 200:
                        return response.json()
        except requests.RequestException:
                return None
        return None


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data_stats():
    df = pd.read_csv(DATA_PATH)

    df["balance"] = pd.to_numeric(df["balance"], errors="coerce")
    medyan_bakiye = df["balance"].median()

    df["marital"] = df["marital"].replace(["EVLI", "singel"], df["marital"].mode()[0])
    df["age"] = df["age"].fillna(df["age"].median())
    df["education"] = df["education"].fillna(df["education"].mode()[0])
    df["housing"] = df["housing"].fillna(df["housing"].mode()[0])
    df["job"] = df["job"].replace("unknown", df["job"].mode()[0])
    df["education"] = df["education"].replace("unknown", df["education"].mode()[0])
    df["poutcome"] = df["poutcome"].replace("unknown", "Ilk_Kez_Araniyor")

    stats = {
        "medyan_bakiye": medyan_bakiye,
        "mode_marital": df["marital"].mode()[0],
        "mode_job": df["job"].mode()[0],
        "mode_education": df["education"].mode()[0],
        "mode_housing": df["housing"].mode()[0],
        "mode_default": df["default"].mode()[0],
        "mode_loan": df["loan"].mode()[0],
        "mode_contact": df["contact"].mode()[0],
        "mode_poutcome": df["poutcome"].mode()[0],
    }

    categories = {
        "job": sorted(df["job"].dropna().unique().tolist()),
        "marital": sorted(df["marital"].dropna().unique().tolist()),
        "education": sorted(df["education"].dropna().unique().tolist()),
        "default": sorted(df["default"].dropna().unique().tolist()),
        "housing": sorted(df["housing"].dropna().unique().tolist()),
        "loan": sorted(df["loan"].dropna().unique().tolist()),
        "contact": sorted(df["contact"].dropna().unique().tolist()),
        "month": ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],
        "poutcome": sorted(df["poutcome"].dropna().unique().tolist()),
    }

    return stats, categories


def preprocess_input(raw, stats):
    data = raw.copy()

    if pd.isna(data.get("balance")):
        data["balance"] = stats["medyan_bakiye"]

    if data.get("marital") in ["EVLI", "singel", None, ""]:
        data["marital"] = stats["mode_marital"]

    if data.get("job") in ["unknown", None, ""]:
        data["job"] = stats["mode_job"]

    if data.get("education") in ["unknown", None, ""]:
        data["education"] = stats["mode_education"]

    if data.get("housing") in [None, ""]:
        data["housing"] = stats["mode_housing"]

    if data.get("default") in [None, ""]:
        data["default"] = stats["mode_default"]

    if data.get("loan") in [None, ""]:
        data["loan"] = stats["mode_loan"]

    if data.get("contact") in [None, ""]:
        data["contact"] = stats["mode_contact"]

    if data.get("poutcome") in ["unknown", None, ""]:
        data["poutcome"] = stats["mode_poutcome"]

    df_input = pd.DataFrame([data])
    df_input = pd.get_dummies(df_input, drop_first=True)
    return df_input


def clean_dataframe(df, stats):
    cleaned = df.copy()
    cleaned["balance"] = pd.to_numeric(cleaned.get("balance"), errors="coerce")
    cleaned["balance"] = cleaned["balance"].fillna(stats["medyan_bakiye"])

    cleaned["marital"] = cleaned["marital"].replace(["EVLI", "singel"], stats["mode_marital"])
    cleaned["age"] = cleaned["age"].fillna(cleaned["age"].median())
    cleaned["education"] = cleaned["education"].fillna(stats["mode_education"])
    cleaned["housing"] = cleaned["housing"].fillna(stats["mode_housing"])

    cleaned["job"] = cleaned["job"].replace("unknown", stats["mode_job"])
    cleaned["education"] = cleaned["education"].replace("unknown", stats["mode_education"])
    cleaned["poutcome"] = cleaned["poutcome"].replace("unknown", stats["mode_poutcome"])
    return cleaned


def align_features(df_input, model):
    if hasattr(model, "feature_names_in_"):
        return df_input.reindex(columns=model.feature_names_in_, fill_value=0)
    return df_input


def get_class_index(model):
    class_index = 1
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if 1 in classes:
            class_index = classes.index(1)
        elif "yes" in classes:
            class_index = classes.index("yes")
    return class_index


def render_gauge(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": COLOR_YES if prob >= 0.5 else COLOR_NO},
                "steps": [
                    {"range": [0, 50], "color": COLOR_NO},
                    {"range": [50, 100], "color": COLOR_YES},
                ],
            },
        )
    )
    fig.update_layout(margin={"t": 10, "b": 10, "l": 10, "r": 10}, height=220)
    st.plotly_chart(fig, use_container_width=True)


def predict_from_raw(raw_input, model, stats):
    df_input = preprocess_input(raw_input, stats)
    df_input = align_features(df_input, model)
    prediction = model.predict(df_input)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        class_index = get_class_index(model)
        proba = float(model.predict_proba(df_input)[0][class_index])
    return prediction, proba


@st.cache_data
def load_feature_matrix():
    df = pd.read_csv(DATA_PATH)
    stats, _ = load_data_stats()
    df = clean_dataframe(df, stats)

    if "deposit" in df.columns:
        y = df["deposit"].replace({"yes": 1, "no": 0})
        X = df.drop(columns=["deposit"])
    else:
        y = None
        X = df.copy()

    X = pd.get_dummies(X, drop_first=True)
    return X, y


@st.cache_resource
def load_shap_explainer():
    model = load_model()
    return shap.TreeExplainer(model)


def get_shap_values(explainer, input_df):
    shap_values = explainer.shap_values(input_df)
    if isinstance(shap_values, list):
        return shap_values[1]
    return shap_values


def build_report_text(focus, summary, insights):
    header = f"# Yonetici Raporu ({focus})\n"
    body = "\n".join(summary + insights)
    return f"{header}\n{body}\n"


def build_pdf_report(text):
    safe_text = unicodedata.normalize("NFKD", text)
    safe_text = safe_text.encode("latin-1", errors="ignore").decode("latin-1")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.set_font("Arial", size=12)
    for line in safe_text.split("\n"):
        pdf.multi_cell(0, 8, line)
    return pdf.output(dest="S").encode("latin-1")


def build_prediction_report(raw_input, prediction, proba):
    status = "EVET" if prediction in [1, "yes"] else "HAYIR"
    prob_text = f"%{proba * 100:.1f}" if proba is not None else "Bilinmiyor"
    lines = [
        "# Tahmin Raporu",
        f"- Sonuc: {status}",
        f"- Guven Skoru: {prob_text}",
        "- Ozet Girdiler:",
        f"  - Yas: {raw_input['age']}",
        f"  - Bakiye: {raw_input['balance']}",
        f"  - Gorusme Suresi: {raw_input['duration']} sn",
        f"  - Meslek: {raw_input['job']}",
        f"  - Medeni Durum: {raw_input['marital']}",
    ]
    return "\n".join(lines)


@st.cache_data
def load_metrics():
    try:
        model = load_model()
    except FileNotFoundError:
        return None

    df = pd.read_csv(DATA_PATH)
    stats, _ = load_data_stats()
    df = clean_dataframe(df, stats)

    if "deposit" not in df.columns:
        return None

    y = df["deposit"].replace({"yes": 1, "no": 0})
    X = df.drop(columns=["deposit"])
    X = pd.get_dummies(X, drop_first=True)
    X = align_features(X, model)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {"accuracy": accuracy, "f1": f1}


st.markdown(
    """
<div class="hero">
  <h1>Banka Musteri Davranisi Analizi</h1>
  <p>Bu uygulama, musterinin vadeli mevduat acma olasiligini tahmin etmek icin tasarlandi.</p>
  <p class="muted"><strong>Business Value:</strong> Dogru hedefleme ile kampanya maliyetini azaltir, donusumu artirir.</p>
</div>
""",
    unsafe_allow_html=True,
)

stats, categories = load_data_stats()
metrics = load_metrics()

kpi_cols = st.columns(4)
try:
    df_stats = pd.read_csv(DATA_PATH)
    total_customers = int(df_stats.shape[0])
    avg_balance = pd.to_numeric(df_stats["balance"], errors="coerce").mean()
except Exception:
    total_customers = 0
    avg_balance = 0

with kpi_cols[0]:
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-title">Toplam Musteri</div>
  <div class="kpi-value">{total_customers:,}</div>
  <div class="muted">👥 Portfoy</div>
</div>
""",
        unsafe_allow_html=True,
    )
with kpi_cols[1]:
    acc_val = f"{metrics['accuracy'] * 100:.2f}%" if metrics else "-"
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-title">Model Accuracy</div>
  <div class="kpi-value">{acc_val}</div>
  <div class="muted">✅ Basari</div>
</div>
""",
        unsafe_allow_html=True,
    )
with kpi_cols[2]:
    f1_val = f"{metrics['f1'] * 100:.2f}%" if metrics else "-"
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-title">F1-Score</div>
  <div class="kpi-value">{f1_val}</div>
  <div class="muted">🎯 Denge</div>
</div>
""",
        unsafe_allow_html=True,
    )
with kpi_cols[3]:
    st.markdown(
        f"""
<div class="kpi">
  <div class="kpi-title">Ortalama Bakiye</div>
  <div class="kpi-value">{avg_balance:,.0f}</div>
  <div class="muted">💳 Ortalama</div>
</div>
""",
        unsafe_allow_html=True,
    )

metric_col1, metric_col2 = st.columns(2)
with metric_col1:
    if metrics:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
    else:
        st.metric("Accuracy", "-")
with metric_col2:
    if metrics:
        st.metric("F1-Score", f"{metrics['f1'] * 100:.2f}%")
    else:
        st.metric("F1-Score", "-")

def reset_form():
    defaults = {
        "age": 35,
        "job": categories["job"][0],
        "marital": categories["marital"][0],
        "education": categories["education"][0],
        "balance": float(stats["medyan_bakiye"]),
        "default": categories["default"][0],
        "housing": categories["housing"][0],
        "loan": categories["loan"][0],
        "contact": categories["contact"][0],
        "day": 15,
        "month": categories["month"][0],
        "duration": 180,
        "campaign": 2,
        "pdays": -1,
        "previous": 0,
        "poutcome": categories["poutcome"][0],
    }
    for key, value in defaults.items():
        st.session_state[key] = value


st.sidebar.header("Musteri Bilgileri")
st.sidebar.markdown("---")
with st.sidebar.form("input_form"):
    with st.expander("Demografik Bilgiler", expanded=True):
        age = st.slider("Yas", 18, 100, 35, key="age", help="Musterinin yasi (18-100 arasi).")
        job = st.selectbox("Is", categories["job"], index=0, key="job", help="Meslek bilgisi.")
        marital = st.selectbox(
            "Medeni Durum",
            categories["marital"],
            index=0,
            key="marital",
            help="Medeni durum bilgisi.",
        )
        education = st.selectbox(
            "Egitim",
            categories["education"],
            index=0,
            key="education",
            help="Egitim seviyesi.",
        )

    with st.expander("Mali Durum", expanded=True):
        balance = st.number_input(
            "Bakiye",
            min_value=0.0,
            value=float(stats["medyan_bakiye"]),
            key="balance",
            help="Negatif deger girilemez.",
        )
        default = st.selectbox(
            "Kredi Temerrudu",
            categories["default"],
            index=0,
            key="default",
            help="Kredi temerrudu bilgisi.",
        )
        housing = st.selectbox(
            "Konut Kredisi",
            categories["housing"],
            index=0,
            key="housing",
            help="Konut kredisi durumu.",
        )
        loan = st.selectbox(
            "Bireysel Kredi",
            categories["loan"],
            index=0,
            key="loan",
            help="Bireysel kredi durumu.",
        )

    with st.expander("Kampanya Bilgileri", expanded=False):
        contact = st.selectbox(
            "Iletisim Turu",
            categories["contact"],
            index=0,
            key="contact",
            help="Telefon/cep/unknown bilgisi.",
        )
        day = st.number_input(
            "Gorusme Gunu (1-31)",
            min_value=1,
            max_value=31,
            value=15,
            key="day",
            help="Iletisim gunu.",
        )
        month = st.selectbox(
            "Ay",
            categories["month"],
            index=0,
            key="month",
            help="Iletisim ayi.",
        )
        duration = st.slider(
            "Gorusme Suresi (sn)",
            0,
            4000,
            180,
            key="duration",
            help="Gorusme suresi saniye cinsinden.",
        )
        campaign = st.slider(
            "Kampanya Temas Sayisi",
            1,
            50,
            2,
            key="campaign",
            help="Bu kampanyadaki temas sayisi.",
        )
        pdays = st.number_input(
            "Onceki Temastan Gecen Gun",
            value=-1,
            key="pdays",
            help="-1 daha once aranmadi anlamina gelir.",
        )
        previous = st.slider(
            "Onceki Temas Sayisi",
            0,
            50,
            0,
            key="previous",
            help="Onceki kampanya temas sayisi.",
        )
        poutcome = st.selectbox(
            "Onceki Kampanya Sonucu",
            categories["poutcome"],
            index=0,
            key="poutcome",
            help="Onceki kampanya sonucu.",
        )

    submitted = st.form_submit_button("Tahmini Hesapla")

raw_input = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome,
}

tabs = st.tabs(["Tahmin", "XAI", "Yonetici Raporu"])

with tabs[0]:
    if submitted:
        try:
            model = load_model()
            lottie_anim = load_lottie("https://assets9.lottiefiles.com/packages/lf20_tnrzlN.json")
            if lottie_anim:
                st_lottie(lottie_anim, height=140, key="processing")
            with st.spinner("Tahmin hesaplanıyor..."):
                prediction, proba = predict_from_raw(raw_input, model, stats)
            st.balloons()

            col_left, col_right = st.columns([2, 1])
            with col_left:
                reason_text = (
                    "Gorusme suresi ve bakiye degeri, modelin en belirleyici sinyalleri arasindadir. "
                    "Gorusme suresi arttikca EVET olasiligi guclenir."
                )
                if prediction in [1, "yes"]:
                    st.markdown(
                        "<div class='result-yes'><div class='card-title'>Sonuc: EVET</div>"
                        "Musteri mevduat acmaya yatkin.</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='result-no'><div class='card-title'>Sonuc: HAYIR</div>"
                        "Musteri mevduat acmaya yatkin degil.</div>",
                        unsafe_allow_html=True,
                    )
                st.markdown(f"<div class='card muted'>{reason_text}</div>", unsafe_allow_html=True)

            with col_right:
                if proba is not None:
                    st.write("Guven Skoru")
                    render_gauge(min(max(proba, 0.0), 1.0))
                    st.caption(f"Tahmin olasiligi: %{proba * 100:.1f}")
                else:
                    st.info("Model olasilik bilgisi saglamiyor.")

            report_text = build_prediction_report(raw_input, prediction, proba)
            pdf_bytes = build_pdf_report(report_text)
            action_col1, action_col2 = st.columns(2)
            with action_col1:
                if st.button("Yeni Sorgu Baslat"):
                    reset_form()
                    st.experimental_rerun()
            with action_col2:
                st.download_button(
                    label="Sonucu PDF Olarak Indir",
                    data=pdf_bytes,
                    file_name="tahmin_raporu.pdf",
                    mime="application/pdf",
                )

            st.markdown("<br><div class='card'><h3>Senaryo Simulatore (What-If Analysis)</h3></div>", unsafe_allow_html=True)

            sim_col_left, sim_col_right = st.columns([1, 1])

            with sim_col_left:
                st.markdown("<div class='card'><h4>Mevcut Durum</h4></div>", unsafe_allow_html=True)
                if prediction in [1, "yes"]:
                    st.markdown(
                        "<div class='result-yes'>Musteri Mevduat Acmaya Yatkin</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='result-no'>Musteri Mevduat Acmaya Yatkin Degil</div>",
                        unsafe_allow_html=True,
                    )
                if proba is not None:
                    st.caption(f"Mevcut olasilik: %{proba * 100:.1f}")

            with sim_col_right:
                st.markdown("<div class='card'><h4>Simulasyon Paneli</h4></div>", unsafe_allow_html=True)
                sim_duration = st.slider(
                    "Gorusme Suresi (sn)",
                    0,
                    4000,
                    int(duration),
                    key="sim_duration",
                )
                sim_balance = st.slider(
                    "Bakiye",
                    min_value=int(min(0, stats["medyan_bakiye"] - 2000)),
                    max_value=int(stats["medyan_bakiye"] + 10000),
                    value=int(balance),
                    key="sim_balance",
                )

                sim_input = raw_input.copy()
                sim_input["duration"] = sim_duration
                sim_input["balance"] = sim_balance

                sim_pred, sim_proba = predict_from_raw(sim_input, model, stats)

                if sim_pred in [1, "yes"]:
                    st.markdown(
                        "<div class='result-yes'>Simulasyon: Musteri Mevduat Acmaya Yatkin</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='result-no'>Simulasyon: Musteri Mevduat Acmaya Yatkin Degil</div>",
                        unsafe_allow_html=True,
                    )
                if sim_proba is not None:
                    st.caption(f"Simulasyon olasiligi: %{sim_proba * 100:.1f}")

            if prediction in [0, "no"] and sim_pred in [1, "yes"]:
                delta_seconds = max(sim_duration - int(duration), 0)
                delta_minutes = max(delta_seconds // 60, 1)
                st.success(
                    f"Kritik Esik Gecildi: Bu musteri {delta_minutes} dakika daha fazla konusursa ikna edilebilir!"
                )

            if proba is not None and sim_proba is not None:
                chart_col_left, chart_col_right = st.columns([2, 1])
                with chart_col_left:
                    duration_range = np.linspace(0, 4000, 50).astype(int)
                    prob_values = []
                    for value in duration_range:
                        temp_input = raw_input.copy()
                        temp_input["duration"] = int(value)
                        _, temp_proba = predict_from_raw(temp_input, model, stats)
                        prob_values.append(0 if temp_proba is None else temp_proba)
                    chart_df = pd.DataFrame(
                        {
                            "duration": duration_range,
                            "probability": prob_values,
                        }
                    )
                    st.line_chart(chart_df, x="duration", y="probability")
                with chart_col_right:
                    st.markdown("<div class='card'><h4>Yorum</h4></div>", unsafe_allow_html=True)
                    st.write("Gorusme suresi arttikca EVET olasiligi yukselir.")

        except FileNotFoundError:
            st.error("Model dosyasi bulunamadi. Lutfen once modeli kaydedin (model.joblib).")
        except Exception as exc:
            st.warning(f"Beklenmeyen bir hata olustu: {exc}")
    else:
        st.markdown("<div class='card muted'>Sol panelden bilgileri girip Tahmini Hesapla butonuna basiniz.</div>", unsafe_allow_html=True)

    st.markdown("<br><div class='card'><h3 class='card-title'>Toplu Tahmin (Batch Processing)</h3></div>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("CSV dosyasi yukle", type=["csv"])

    if uploaded_file is not None:
        try:
            model = load_model()
            batch_df = pd.read_csv(uploaded_file)

            required_cols = [
                "age",
                "job",
                "marital",
                "education",
                "default",
                "balance",
                "housing",
                "loan",
                "contact",
                "day",
                "month",
                "duration",
                "campaign",
                "pdays",
                "previous",
                "poutcome",
            ]

            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            if missing_cols:
                st.warning(
                    f"Eksik kolonlar tespit edildi: {', '.join(missing_cols)}. Varsayilan degerler kullanilacak."
                )
                for col in missing_cols:
                    batch_df[col] = np.nan

            batch_df = batch_df[required_cols]
            batch_df = clean_dataframe(batch_df, stats)

            batch_features = pd.get_dummies(batch_df, drop_first=True)
            batch_features = align_features(batch_features, model)

            batch_pred = model.predict(batch_features)
            batch_proba = None
            if hasattr(model, "predict_proba"):
                class_index = get_class_index(model)
                batch_proba = model.predict_proba(batch_features)[:, class_index]

            output_df = batch_df.copy()
            output_df["Tahmin"] = ["YES" if val in [1, "yes"] else "NO" for val in batch_pred]
            if batch_proba is not None:
                output_df["Olasilik"] = (batch_proba * 100).round(2)
            else:
                output_df["Olasilik"] = np.nan

            st.dataframe(output_df, use_container_width=True)

            csv_data = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Tahminleri indir (tahminler.csv)",
                data=csv_data,
                file_name="tahminler.csv",
                mime="text/csv",
            )

        except FileNotFoundError:
            st.error("Model dosyasi bulunamadi. Lutfen once modeli kaydedin (model.joblib).")
        except Exception as exc:
            st.warning(f"Batch tahmin sirasinda hata olustu: {exc}")

with tabs[1]:
    st.markdown("<div class='card'><h3>Model Seffafligi ve Aciklanabilirlik (XAI)</h3></div>", unsafe_allow_html=True)
    try:
        model = load_model()
        X, y = load_feature_matrix()
        X = align_features(X, model)

        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame(
                {
                    "feature": X.columns,
                    "importance": model.feature_importances_,
                }
            ).sort_values(by="importance", ascending=False)
            fig = go.Figure(
                go.Bar(
                    x=fi["importance"].head(15),
                    y=fi["feature"].head(15),
                    orientation="h",
                    marker_color=COLOR_YES,
                )
            )
            fig.update_layout(
                title="Feature Importance (Top 15)",
                height=400,
                margin={"l": 10, "r": 10, "t": 40, "b": 10},
                yaxis={"autorange": "reversed"},
            )
            st.plotly_chart(fig, use_container_width=True)

        if submitted:
            df_input = preprocess_input(raw_input, stats)
            df_input = align_features(df_input, model)

            explainer = load_shap_explainer()
            shap_values = get_shap_values(explainer, df_input)

            st.markdown("<div class='card'><h4>Lokal Aciklanabilirlik (SHAP)</h4></div>", unsafe_allow_html=True)
            shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[1], data=df_input.iloc[0], feature_names=df_input.columns), show=False)
            st.pyplot(bbox_inches="tight")

            abs_shap = np.abs(shap_values[0])
            top_idx = np.argsort(abs_shap)[-3:][::-1]
            top_features = df_input.columns[top_idx].tolist()
            st.caption(
                "Bu musteri icin en belirleyici faktorler: " + ", ".join(top_features) + "."
            )
        else:
            st.info("Lokal aciklama icin once bir musteri tahmini yapin.")
    except FileNotFoundError:
        st.error("Model dosyasi bulunamadi. Lutfen once modeli kaydedin (model.joblib).")
    except Exception as exc:
        st.warning(f"XAI hesaplamasi sirasinda hata olustu: {exc}")

with tabs[2]:
    st.markdown("<div class='card'><h3>Yonetici Raporu Olusturma</h3></div>", unsafe_allow_html=True)
    focus = st.selectbox(
        "Raporun odak noktasi ne olsun?",
        ["Satis Stratejisi", "Risk Analizi", "Operasyonel Verimlilik"],
    )

    summary = [
        "## Genel Ozet",
        "- Model kampanya hedeflemesini iyilestirmek icin kullanilir.",
    ]
    insights = []

    if focus == "Satis Stratejisi":
        insights.extend(
            [
                "## Pazarlama Muduru Icin",
                "- Donusum oranini artirmak icin gorusme suresini artirin.",
                "- En verimli iletisim donemlerini hedefleyin.",
            ]
        )
    elif focus == "Risk Analizi":
        insights.extend(
            [
                "## Risk Analizi",
                "- Temerrut ve kredi yukleri risk profilini belirler.",
                "- Riskli gruplar icin ek dogrulama uygulanmali.",
            ]
        )
    else:
        insights.extend(
            [
                "## Operasyonel Verimlilik",
                "- Arama sayisi arttikca kabul orani dusmektedir.",
                "- Dogru segmentasyon, maliyeti azaltir.",
            ]
        )

    try:
        X, y = load_feature_matrix()
        df_raw = pd.read_csv(DATA_PATH)
        df_raw = clean_dataframe(df_raw, stats)
        duration_q1 = df_raw["duration"].quantile(0.25)
        duration_q3 = df_raw["duration"].quantile(0.75)
        iqr = duration_q3 - duration_q1
        upper = duration_q3 + 1.5 * iqr
        outlier_count = int((df_raw["duration"] > upper).sum())

        insights.append(f"- Aykiri gorusme suresi sayisi: {outlier_count}.")

        model = load_model()
        if getattr(model, "max_depth", None) == 7:
            insights.append("- max_depth=7 budamasi modelin genelleme gucunu artirdi.")
        else:
            insights.append("- max_depth siniri yok; budama (max_depth=7) robustness icin onerilir.")

        if y is not None:
            df_temp = df_raw.copy()
            df_temp["deposit"] = df_temp["deposit"].replace({"yes": 1, "no": 0})
            rate_by_month = df_temp.groupby("month")["deposit"].mean().reindex(
                ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
            )
            st.bar_chart(rate_by_month)
    except Exception:
        pass

    report_text = build_report_text(focus, summary, insights)
    st.markdown(report_text)

    st.download_button(
        label="Raporu indir (Markdown)",
        data=report_text.encode("utf-8"),
        file_name="rapor.md",
        mime="text/markdown",
    )

    pdf_bytes = build_pdf_report(report_text)
    st.download_button(
        label="Raporu indir (PDF)",
        data=pdf_bytes,
        file_name="rapor.pdf",
        mime="application/pdf",
    )
