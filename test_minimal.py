"""Minimal Streamlit test - just model selector + API key"""
import streamlit as st

st.set_page_config(page_title="Test", layout="wide")

st.sidebar.title("Test Sidebar")

# Simple selectbox
st.sidebar.markdown("### Model")
provider = st.sidebar.selectbox(
    "Provider",
    options=["ollama", "deepseek", "openai"],
    index=0,
    key="test_provider",
)
st.sidebar.write(f"Selected: **{provider}**")

# Show key input for cloud
if provider != "ollama":
    label = "DeepSeek" if provider == "deepseek" else "OpenAI"
    key_val = st.sidebar.text_input(
        f"{label} API Key",
        type="password",
        placeholder="sk-...",
        key="test_key",
    )
    if key_val:
        st.sidebar.success(f"Key: {key_val[:6]}...{key_val[-4:]}")
    else:
        st.sidebar.warning("Enter key above")
else:
    st.sidebar.success("Local mode - no key needed")

st.sidebar.markdown("---")
st.sidebar.caption("Test v1")

st.title("Minimal Test")
st.write("If you see this, the basic UI works.")

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("Content 1")
with tab2:
    st.write("Content 2")
