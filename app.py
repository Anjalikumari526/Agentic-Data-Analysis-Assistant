
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import importlib

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

# ---------------- SESSION STATE ----------------
if "df" not in st.session_state:
    st.session_state.df = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("📊 Data Assistant")
st.sidebar.caption("Agentic Data Analysis Platform")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Data Explorer", "AI Data Analyst", "Analysis Results"]
)

st.sidebar.divider()
st.sidebar.subheader("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.sidebar.success("Dataset loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")

df = st.session_state.df

# ---------------- COMMON FUNCTIONS ----------------
def require_data():
    if df is None:
        st.warning("Please upload a CSV dataset from the sidebar.")
        return False
    return True


def show_metrics(data):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(data):,}")
    c2.metric("Total Columns", len(data.columns))
    c3.metric("Missing Values", int(data.isna().sum().sum()))
    c4.metric(
        "Duplicate Rows",
        int(data.duplicated().sum())
    )


def create_chart(data):
    numeric = data.select_dtypes(include=np.number).columns.tolist()
    categorical = data.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    if numeric:
        col = st.selectbox("Select numerical column", numeric)
        fig = px.histogram(
            data,
            x=col,
            title=f"Distribution of {col}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    if categorical:
        col = st.selectbox(
            "Select categorical column",
            categorical,
            key="category_chart"
        )

        counts = data[col].value_counts().head(15)

        fig = px.bar(
            x=counts.index.astype(str),
            y=counts.values,
            labels={"x": col, "y": "Count"},
            title=f"Distribution of {col}",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)


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
                "Missing Values": df.isna().sum().values
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
                lambda col: col.str.contains(
                    search, case=False, na=False
                )
            ).any(axis=1)
            filtered = filtered[mask]

        col1, col2 = st.columns(2)

        with col1:
            selected = st.selectbox(
                "Filter by column",
                ["None"] + list(df.columns)
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
            mime="text/csv"
        )


# ==================================================
# 3. AI DATA ANALYST
# ==================================================
elif page == "AI Data Analyst":

    st.header("🤖 AI Data Analyst")
    st.caption(
        "Ask questions about your dataset using our AI agent."
    )

    if require_data():

        st.info(
            "Example: What is the average sales value? "
            "Which region performs best?"
        )

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input(
            "Ask anything about your dataset..."
        )

        if question:

            st.session_state.chat_history.append({
                "role": "user",
                "content": question
            })

            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):

                try:
                    # Import teammate's agent when available
                    agent_module = importlib.import_module(
                        "agent.agent"
                    )

                    analyze_query = getattr(
                        agent_module, "analyze_query"
                    )

                    with st.spinner("AI agent is analyzing your data..."):
                        result = analyze_query(df.copy(), question)

                    if isinstance(result, dict):
                        answer = result.get(
                            "answer",
                            "Analysis completed."
                        )

                        st.markdown(answer)

                        for chart in result.get("charts", []):
                            st.plotly_chart(
                                chart,
                                use_container_width=True
                            )

                        for table in result.get("tables", []):
                            st.dataframe(
                                table,
                                use_container_width=True
                            )

                    else:
                        answer = str(result)
                        st.markdown(answer)

                    st.session_state.analysis_results.append({
                        "question": question,
                        "answer": answer
                    })

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer
                    })

                except ModuleNotFoundError:
                    st.warning(
                        "AI agent is not connected yet. "
                        "The frontend is ready for integration."
                    )

                    st.code(
                        "def analyze_query(df, question):\n"
                        "    # Your teammate's agent logic\n"
                        "    return {\n"
                        '        "answer": "Your analysis",\n'
                        '        "charts": [],\n'
                        '        "tables": []\n'
                        "    }",
                        language="python"
                    )

                except Exception as e:
                    st.error(f"Agent error: {e}")


# ==================================================
# 4. ANALYSIS RESULTS
# ==================================================
elif page == "Analysis Results":

    st.header("📋 Analysis Results")

    if not require_data():
        pass

    else:
        show_metrics(df)

        st.subheader("Statistical Summary")

        st.dataframe(
            df.describe(include="all").T,
            use_container_width=True
        )

        st.subheader("Column Information")

        info = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Missing": df.isna().sum().values,
            "Unique Values": df.nunique().values
        })

        st.dataframe(info, use_container_width=True)

        st.divider()
        st.subheader("AI Analysis History")

        if st.session_state.analysis_results:
            st.dataframe(
                pd.DataFrame(st.session_state.analysis_results),
                use_container_width=True
            )
        else:
            st.info("No AI analysis has been performed yet.")

        st.download_button(
            "Download Dataset Report",
            df.describe(include="all").to_csv().encode("utf-8"),
            file_name="analysis_report.csv",
            mime="text/csv"
        )

# ---------------- FOOTER ----------------
st.sidebar.divider()
st.sidebar.caption("Agentic Data Analysis Assistant")
st.sidebar.caption("Built with Python and Streamlit")
