import operator
from typing import TypedDict, Annotated
from pydantic import BaseModel, Field, field_validator
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END

# Model setup
model = ChatOllama(
    model="qwen2.5:1.5b",
    temperature=0
)

# 1. FIX: Flexible Field Definition + Strict Pydantic Validator
class EssayEvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback on the essay aspect")
    score: int = Field(description="Score out of 10 between 1 and 10")

    # Mode 'before' safely captures raw LLM output before type enforcement
    @field_validator('score', mode='before')
    @classmethod
    def clean_and_parse_score(cls, v):
        # Case 1: If string output comes from LLM
        if isinstance(v, str):
            try:
                # Extract numeric digits if present (e.g., "8/10" or "8")
                digits = "".join([char for char in v if char.isdigit()])
                val = int(digits) if digits else 5
            except ValueError:
                val = 5  # Fallback score if completely non-numeric (e.g. "sync")
        elif isinstance(v, (int, float)):
            val = int(v)
        else:
            val = 5

        # Clamp value strictly between 1 and 10
        return max(1, min(10, val))

# 2. Schema Binding
structured_model = model.with_structured_output(EssayEvaluationSchema)

# State Definition
class UPSCState(TypedDict):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[list[int], operator.add]
    avg_score: float

# Node Functions
def evaluate_language(state: UPSCState):
    prompt = f"Evaluate the language quality of the following essay. Return feedback and a numerical score from 1 to 10.\nEssay:\n{state['essay']}" 
    output = structured_model.invoke(prompt)
    return {'language_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_analysis(state: UPSCState):
    prompt = f"Evaluate the depth of analysis of the following essay. Return feedback and a numerical score from 1 to 10.\nEssay:\n{state['essay']}" 
    output = structured_model.invoke(prompt)
    return {'analysis_feedback': output.feedback, 'individual_scores': [output.score]}

def evaluate_thought(state: UPSCState):
    prompt = f"Evaluate the clarity of thought of the following essay. Return feedback and a numerical score from 1 to 10.\nEssay:\n{state['essay']}" 
    output = structured_model.invoke(prompt)
    return {'clarity_feedback': output.feedback, 'individual_scores': [output.score]}

def final_evaluation(state: UPSCState):
    prompt = f"""Based on the following feedbacks, create a summarized overall feedback:
Language Feedback: {state.get('language_feedback', '')}
Analysis Feedback: {state.get('analysis_feedback', '')}
Clarity of Thought Feedback: {state.get('clarity_feedback', '')}""" 
    
    overall_feedback = model.invoke(prompt).content

    # Average score calculation
    scores = state.get('individual_scores', [])
    avg_score = sum(scores) / len(scores) if scores else 0.0

    return {'overall_feedback': overall_feedback, 'avg_score': avg_score}

# Graph Construction
graph = StateGraph(UPSCState)

graph.add_node('evaluate_language', evaluate_language)
graph.add_node('evaluate_analysis', evaluate_analysis)
graph.add_node('evaluate_thought', evaluate_thought)
graph.add_node('final_evaluation', final_evaluation)

# Edges
graph.add_edge(START, 'evaluate_language')
graph.add_edge(START, 'evaluate_analysis')
graph.add_edge(START, 'evaluate_thought')

graph.add_edge('evaluate_language', 'final_evaluation')
graph.add_edge('evaluate_analysis', 'final_evaluation')
graph.add_edge('evaluate_thought', 'final_evaluation')

graph.add_edge('final_evaluation', END)

workflow = graph.compile()

# Execution
essay_input = """**The Importance of Time**
Time is one of the most valuable assets in human life. Unlike money or material possessions, once time is lost, it can never be recovered. Every second that passes brings new opportunities to learn, grow, and achieve our goals.
Effective time management allows individuals to balance work, studies, and personal life efficiently. People who value time and remain disciplined are more likely to achieve success and peace of mind. Conversely, wasting time leads to missed opportunities and regret. Therefore, respecting time and using it wisely is essential for leading a meaningful, productive, and successful life."""

initial_state = {
    'essay': essay_input
}

result = workflow.invoke(initial_state)

print("\n=== FINAL RESULTS ===")
print("Individual Scores:", result.get('individual_scores'))
print("Average Score:", result.get('avg_score'))
print("\nOverall Feedback:\n", result.get('overall_feedback'))