from langchain_community.llms import Ollama


llm = Ollama(model="llama3")


def compare_candidates(candidates):

    prompt = f"""
    Compare these candidates side by side.

    Candidates:
    {candidates}

    Compare:
    - skills
    - experience
    - strengths
    - weaknesses
    - best fit

    Give detailed comparison.
    """

    response = llm.invoke(prompt)

    return response


def generate_interview_questions(candidate):

    prompt = f"""
    Generate interview questions for this candidate.

    Include:
    - technical questions
    - HR questions
    - project questions

    Candidate:
    {candidate}
    """

    response = llm.invoke(prompt)

    return response


def explain_ranking(candidates):

    prompt = f"""
    Explain why the top candidates ranked higher.

    Focus on:
    - technical skills
    - experience
    - project relevance
    - strengths
    - weaknesses

    Candidates:
    {candidates}
    """

    response = llm.invoke(prompt)

    return response