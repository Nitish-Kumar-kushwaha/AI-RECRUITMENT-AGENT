import streamlit as st

from matching_agent import graph

from tools import (
    compare_candidates,
    generate_interview_questions,
    explain_ranking
)

st.set_page_config(
    page_title="AI Recruitment Agent",
    layout="wide"
)

st.title("🤖 AI Recruitment Agent")

st.write("Find and rank candidates using AI")

query = st.text_input(
    "Enter Hiring Requirement"
)

result = None

if st.button("Search Candidates"):

    with st.spinner("Searching candidates..."):

        result = graph.invoke({

            "user_query": query,

            "conversation_history": []
        })

        st.subheader("📋 Hiring Report")

        st.write(result["final_report"])


if result:

    if st.button("Compare Top Candidates"):

        comparison = compare_candidates(
            result["ranked_candidates"]
        )

        st.subheader("📊 Candidate Comparison")

        st.write(comparison)


    if st.button("Generate Interview Questions"):

        questions = generate_interview_questions(
            result["ranked_candidates"]
        )

        st.subheader("🎤 Interview Questions")

        st.write(questions)


    if st.button("Explain Candidate Rankings"):

        explanation = explain_ranking(
            result["ranked_candidates"]
        )

        st.subheader("🧠 Ranking Explanation")

        st.write(explanation)