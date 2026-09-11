import uuid
import requests
import streamlit as st


API_URL = "http://localhost:8000"


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


st.set_page_config(
    page_title="HDFC Loan Intelligence",
    page_icon="🏦",
    layout="wide",
)


st.title("🏦 HDFC Loan Intelligence")

st.caption(
    "AI-assisted loan analysis, policy guidance and decision review"
)


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


query = st.chat_input(
    "Ask about a loan..."
)


if query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": query,
        }
    )

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = requests.post(
                f"{API_URL}/api/v1/loan/workflow",
                json={
                    "session_id": st.session_state.session_id,
                    "query": query,
                },
                timeout=120,
            )

        if response.ok:
            result = response.json()

            # Render final response
            answer = result.get(
                "final_answer",
                "No response generated."
            )

            st.markdown(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )

        else:
            answer = f"Error: {response.text}"

            st.error(answer)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )