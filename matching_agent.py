from typing import TypedDict, List

from langgraph.graph import StateGraph, END

from langchain_community.llms import Ollama

from rag_pipeline import get_retriever

from dotenv import load_dotenv

load_dotenv()



llm = Ollama(model="llama3")



retriever = get_retriever()



class AgentState(TypedDict):

    user_query: str

    retrieved_candidates: list

    ranked_candidates: str

    final_report: str

    conversation_history: List[str]





def search_candidates(state):

    results = retriever.invoke(
        state["user_query"]
    )

    return {
        **state,
        "retrieved_candidates": results
    }





def rank_candidates(state):

    prompt = f"""
    You are an AI recruitment assistant.

    User Requirement:
    {state["user_query"]}

    Candidate Resumes:
    {state["retrieved_candidates"]}

    Rank the candidates based on:
    - skills
    - experience
    - relevance

    Explain why each candidate ranked where they did.
    """

    response = llm.invoke(prompt)

    return {
        **state,
        "ranked_candidates": response
    }





def generate_report(state):

    prompt = f"""
    Create a hiring report for recruiter.

    Include:
    - top candidates
    - strengths
    - weaknesses
    - hiring recommendation

    Ranked Candidates:
    {state["ranked_candidates"]}
    """

    response = llm.invoke(prompt)

    return {
        **state,
        "final_report": response
    }




workflow = StateGraph(AgentState)

workflow.add_node(
    "search_candidates",
    search_candidates
)

workflow.add_node(
    "rank_candidates",
    rank_candidates
)

workflow.add_node(
    "generate_report",
    generate_report
)

workflow.set_entry_point(
    "search_candidates"
)

workflow.add_edge(
    "search_candidates",
    "rank_candidates"
)

workflow.add_edge(
    "rank_candidates",
    "generate_report"
)

workflow.add_edge(
    "generate_report",
    END
)

graph = workflow.compile()




if __name__ == "__main__":

    query = input("Enter Hiring Requirement: ")

    result = graph.invoke({

        "user_query": query,

        "conversation_history": []
    })

    print("\nFINAL REPORT:\n")

    print(result["final_report"])