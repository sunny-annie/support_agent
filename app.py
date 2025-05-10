import streamlit as st
from agent_conditions import run_agent

st.set_page_config(page_title="🤖 AI-помощник службы поддержки")
st.title("💬 AI-помощник службы поддержки")

with st.form("support_form"):
    user_query = st.text_area("Введите вопрос или обращение:", height=200)
    submitted = st.form_submit_button("Отправить")

if submitted and user_query.strip():
    with st.spinner("Обработка запроса..."):
        response = run_agent(user_query)
    st.markdown("### 📍 Ответ:")
    st.success(response)