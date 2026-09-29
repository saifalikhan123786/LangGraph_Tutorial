from typing import Annotated, Literal, TypedDict
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field, field_validator

## Model
model = ChatOllama(model="qwen2.5:1.5b", temperature=0)


## Class
class SentimentSchema(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(
        description="Sentiment of the review"
    )

class DiagnosisSchema(BaseModel):
    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')
    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')


# Model
structured_model = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)



prompt = "What is the sentimemnt of the following review - The software too bad"
result = structured_model.invoke(prompt)
print(result)


# State
class ReviewState(TypedDict):
    review: str
    sentiment: Literal["positive", "negative"]
    diagnosis: dict
    response: str


# Function (Class ke bahar)
def find_sentiment(state: ReviewState):
    prompt = f'For the following review find out the sentiment \n {state["review"]}'
    sentiment = structured_model.invoke(
        prompt
    ).sentiment  # Double .invoke() hata kar sentiment property read ki hai

    return {"sentiment": sentiment}


def check_sentiment(state:ReviewState)-> Literal["positive_response","run_diagnosis"]:
    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'


def positive_response(state: ReviewState):
    prompt = f"""Write a warm thank_you message in response to this review:
    
"{state['review']}"

Also, kindly ask the user to leave feedback on your website."""
    response=model.invoke(prompt).content
    return{'response':response}


def run_diagnosis(state: ReviewState):

    prompt = f"""Diagnose this negative review:\n\n{state['review']}\n"
    "Return issue_type, tone, and urgency.
"""
    response=structured_model2.invoke(prompt)
    return{'diagnosis':response.model_dump()}


def negative_response(state: ReviewState):

    diagnosis = state['diagnosis']

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content

    return {'response': response}



# Graph (Class ke bahar)
graph = StateGraph(ReviewState)
graph.add_node("find_sentiment", find_sentiment)
graph.add_node('positive_response',positive_response)
graph.add_node('run_diagnosis',run_diagnosis)
graph.add_node('negative_response',negative_response)



graph.add_edge(START, 'find_sentiment')
graph.add_conditional_edges('find_sentiment',check_sentiment)
graph.add_edge('positive_response',END)
graph.add_edge('run_diagnosis','negative_response')
graph.add_edge('find_sentiment', END)

# Graph compile & execution
workflow = graph.compile()
workflow

initial_state={
    'review':'The product was really good'
}
workflow.invoke(initial_state)
