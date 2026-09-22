import streamlit as st
from api_client import APIClientError, ask_question, get_health

st.set_page_config(
    page_title="Asthma Guideline Assistant",
    page_icon="🫁",
    layout="centered",
)

st.title("Asthma Guideline Assistant")
st.caption("Evidence-grounded answers from indexed GINA, NICE, and NHLBI documents.")
st.info(
    "Educational information only. This assistant does not diagnose conditions "
    "or replace a qualified healthcare professional. For urgent breathing "
    "problems, seek immediate local emergency care."
)

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(message: dict) -> None:
    sources = message.get("sources") or []
    details = message.get("source_details") or []
    if sources:
        st.markdown("**Sources**")
        for source in sources:
            st.markdown(f"- {source}")
    if details:
        with st.expander("Retrieved evidence"):
            for detail in details:
                document = detail.get("document", "Unknown document")
                page_start = detail.get("page_start")
                page_end = detail.get("page_end")
                page = (
                    str(page_start)
                    if page_start == page_end
                    else f"{page_start}–{page_end}"
                )
                st.markdown(f"**{document}, PDF page {page}**")
                if detail.get("section"):
                    st.caption(f"Section: {detail['section']}")
                if detail.get("text_preview"):
                    st.write(detail["text_preview"])
                if detail.get("source_url"):
                    st.markdown(f"[Official source]({detail['source_url']})")


with st.sidebar:
    st.subheader("Service")
    if st.button("Check backend"):
        try:
            health = get_health()
            if health.get("rag_ready"):
                st.success(f"Ready · {health.get('indexed_chunks', 0)} indexed chunks")
            else:
                st.warning(health.get("detail") or "Backend is running but not ready.")
        except APIClientError as exc:
            st.error(str(exc))

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    st.subheader("Try asking")
    st.write("• What is MART therapy?")
    st.write("• How is asthma diagnosed in adults?")
    st.write("• What should be checked at an asthma review?")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message)


if question := st.chat_input("Ask an asthma-guideline question"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with (
        st.chat_message("assistant"),
        st.spinner("Retrieving guideline evidence and checking the answer..."),
    ):
        try:
            response = ask_question(question)
        except APIClientError as exc:
            answer = f"Unable to answer: {exc}"
            response = {"answer": answer, "sources": [], "source_details": []}
            st.error(str(exc))
        else:
            answer = response["answer"]
            st.markdown(answer)
            render_sources(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": response.get("sources", []),
            "source_details": response.get("source_details", []),
        }
    )
