import os
import hashlib

import requests
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Agentic Data Analysis Assistant",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM STYLE ----------------
st.markdown("""
<style>
.main {background-color: #0e1117;}
[data-testid="stSidebar"] {background-color: #161b22;}
.metric-card {
    background: #1c2330;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #30363d;
}
</style>
""", unsafe_allow_html=True)

# ---------------- BACKEND CONFIG ----------------
def get_setting(name, default=""):
    """Read deployment secrets first, then environment variables."""
    try:
        value = st.secrets.get(name, default)
    except Exception:
        value = default
    return os.getenv(name, value)


BACKEND_URL = get_setting("BACKEND_URL").rstrip("/")
API_KEY = get_setting("API_KEY")

# ---------------- SESSION STATE ----------------
defaults = {
    "df": None,
    "chat_history": [],
    "analysis_results": [],
    "uploaded_file_hash": None,
    "backend_upload_info": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ---------------- COMMON FUNCTIONS ----------------
def api_headers():
    return {"Authorization": f"Bearer {API_KEY}"}


def backend_is_configured():
    return bool(BACKEND_URL and API_KEY)


def check_backend():
    if not backend_is_configured():
        return False, (
            "Backend is not configured. Add BACKEND_URL and API_KEY "
            "to Streamlit secrets."
        )
    try:
        response = requests.get(
            f"{BACKEND_URL}/health",
            headers=api_headers(),
            timeout=15,
        )
        response.raise_for_status()
        return True, response.json().get("message", "Backend is online.")
    except requests.RequestException as exc:
        return False, f"Could not reach backend: {exc}"


def upload_to_backend(file_bytes, filename):
    """Send the CSV to the FastAPI /upload endpoint."""
    response = requests.post(
        f"{BACKEND_URL}/upload",
        headers=api_headers(),
        files={"file": (filename, file_bytes, "text/csv")},
        timeout=120,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Dataset upload failed ({response.status_code}): {detail}")
    return response.json()


def ask_backend(question):
    """Send a natural-language question to FastAPI /chat."""
    response = requests.post(
        f"{BACKEND_URL}/chat",
        headers={**api_headers(), "Content-Type": "application/json"},
        json={"question": question},
        timeout=300,
    )
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Agent request failed ({response.status_code}): {detail}")
    return response.json()


def require_data():
    if st.session_state.df is None:
        st.warning("Please upload a CSV dataset from the sidebar.")
        return False
    return True


def show_metrics(data):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", f"{len(data):,}")
    c2.metric("Total Columns", len(data.columns))
    c3.metric("Missing Values", int(data.isna().sum().sum()))
    c4.metric("Duplicate Rows", int(data.duplicated().sum()))


def create_chart(data):
    numeric = data.select_dtypes(include=np.number).columns.tolist()
    categorical = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if numeric:
        col = st.selectbox("Select numerical column", numeric)
        fig = px.histogram(
            data, x=col, title=f"Distribution of {col}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    if categorical:
        col = st.selectbox(
            "Select categorical column", categorical, key="category_chart"
        )
        counts = data[col].value_counts().head(15)
        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={"x": col, "y": "Count"},
            title=f"Distribution of {col}",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True)


# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Data Assistant")
st.sidebar.caption("Agentic Data Analysis Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Explorer", "AI Data Analyst", "Analysis Results"]
)

st.sidebar.divider()
st.sidebar.subheader("Backend Connection")

if backend_is_configured():
    if st.sidebar.button("Check backend connection"):
        with st.sidebar:
            with st.spinner("Checking backend..."):
                ok, message = check_backend()
                if ok:
                    st.success(message)
                else:
                    st.error(message)
else:
    st.sidebar.warning("Backend URL/API key not configured.")

st.sidebar.divider()
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"],
    key="dataset_uploader",
)

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.sha256(file_bytes).hexdigest()

        # Load locally for dashboard/explorer and upload only when changed.
        if file_hash != st.session_state.uploaded_file_hash:
            new_df = pd.read_csv(pd.io.common.BytesIO(file_bytes))
            if new_df.empty:
                raise ValueError("The CSV contains no data rows.")

            if not backend_is_configured():
                st.session_state.df = new_df
                st.session_state.uploaded_file_hash = file_hash
                st.sidebar.warning(
                    "CSV loaded locally. Configure the backend to enable AI analysis."
                )
            else:
                with st.sidebar:
                    with st.spinner("Uploading dataset to AI backend..."):
                        upload_info = upload_to_backend(
                            file_bytes, uploaded_file.name
                        )
                st.session_state.df = new_df
                st.session_state.uploaded_file_hash = file_hash
                st.session_state.backend_upload_info = upload_info
                st.session_state.chat_history = []
                st.session_state.analysis_results = []
                st.sidebar.success(
                    f"Uploaded: {upload_info.get('rows', len(new_df)):,} rows"
                )
    except Exception as exc:
        st.sidebar.error(f"Upload error: {exc}")

df = st.session_state.df

# ---------------- APPLICATION HEADER ----------------
st.title("📊 Agentic Data Analysis Assistant")
st.caption(
    "An intelligent platform for automated data exploration, "
    "visualization and AI-powered insights."
)

# ==================================================
# 1. DASHBOARD
# ==================================================
if page == "Dashboard":
    st.header("📈 Dashboard")

    if require_data():
        show_metrics(df)
        st.divider()

        left, right = st.columns(2)
        with left:
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

        with right:
            st.subheader("Dataset Information")
            info = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Missing Values": df.isna().sum().values,
            })
            st.dataframe(info, use_container_width=True)

        st.divider()
        st.subheader("Data Visualization")
        create_chart(df)

# ==================================================
# 2. DATA EXPLORER
# ==================================================
elif page == "Data Explorer":
    st.header("🔎 Data Explorer")
    st.caption("Search, filter and explore your dataset.")

    if require_data():
        search = st.text_input("Search across all columns")
        filtered = df.copy()

        if search:
            mask = filtered.astype(str).apply(
                lambda col: col.str.contains(search, case=False, na=False)
            ).any(axis=1)
            filtered = filtered[mask]

        col1, col2 = st.columns(2)
        with col1:
            selected = st.selectbox(
                "Filter by column", ["None"] + list(df.columns)
            )
        with col2:
            value = st.text_input("Filter value")

        if selected != "None" and value:
            filtered = filtered[
                filtered[selected].astype(str).str.contains(
                    value, case=False, na=False
                )
            ]

        st.metric("Matching Records", len(filtered))
        st.dataframe(filtered, use_container_width=True)

        st.download_button(
            "⬇️ Download Filtered CSV",
            filtered.to_csv(index=False).encode("utf-8"),
            file_name="filtered_dataset.csv",
            mime="text/csv",
        )

# ==================================================
# 3. AI DATA ANALYST
# ==================================================
elif page == "AI Data Analyst":
    st.header("🤖 AI Data Analyst")
    st.caption("Ask questions about your uploaded dataset using our AI agent.")

    if require_data():
        st.info(
            "Example: What is the average sales value? "
            "Which region performs best?"
        )

        if not backend_is_configured():
            st.warning(
                "The dataset is loaded in the frontend, but the AI backend "
                "is not configured. Add BACKEND_URL and API_KEY to Streamlit secrets."
            )

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "Ask anything about your dataset...",
            disabled=not backend_is_configured(),
        )

        if question:
            st.session_state.chat_history.append({
                "role": "user", "content": question
            })
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                try:
                    with st.spinner("AI agent is analyzing your data..."):
                        result = ask_backend(question)

                    answer = result.get("answer", "The backend returned no answer.")
                    st.markdown(answer)

                    # The current backend returns answer/tool history, not Plotly
                    # chart objects or table objects. Show tool execution details.
                    tool_history = result.get("tool_history", [])
                    if tool_history:
                        with st.expander("View agent tool execution"):
                            for index, item in enumerate(tool_history, start=1):
                                st.markdown(
                                    f"**Step {index}:** `{item.get('tool_name', 'Tool')}`"
                                )
                                st.json(item.get("arguments", {}))
                                st.text(item.get("result", ""))

                    st.session_state.analysis_results.append({
                        "question": question,
                        "answer": answer,
                    })
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": answer
                    })

                except requests.Timeout:
                    st.error(
                        "The AI request timed out. The model may need more time; "
                        "please try again."
                    )
                except Exception as exc:
                    st.error(str(exc))

# ==================================================
# 4. ANALYSIS RESULTS
# ==================================================
elif page == "Analysis Results":
    st.header("📋 Analysis Results")

    if require_data():
        show_metrics(df)
        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        st.subheader("Column Information")
        info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique Values": df.nunique().values,
        })
        st.dataframe(info, use_container_width=True)

        st.divider()
        st.subheader("AI Analysis History")
        if st.session_state.analysis_results:
            st.dataframe(
                pd.DataFrame(st.session_state.analysis_results),
                use_container_width=True,
            )
        else:
            st.info("No AI analysis has been performed yet.")

        st.download_button(
            "Download Dataset Report",
            df.describe(include="all").to_csv().encode("utf-8"),
            file_name="analysis_report.csv",
            mime="text/csv",
        )

# ---------------- FOOTER ----------------
st.sidebar.divider()
st.sidebar.caption("Agentic Data Analysis Assistant")
st.sidebar.caption("Built with Python, Streamlit and FastAPI")
