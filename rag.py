from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

model = ChatOpenAI(openai_api_key="YOUR_API_KEY", temperature=0.3, model="gpt-4o")

def generate_soap_notes(transcription, patient_age, visit_type):
    prompt_template = ChatPromptTemplate.from_template("""
    You're a pediatric clinical assistant generating SOAP notes. 
    Consider patient's age ({age}) and visit type ({visit}).

    Transcript: "{transcription}"

    Use standard pediatric SOAP format (Subjective, Objective, Assessment, Plan).
    """)

    prompt = prompt_template.format_messages(age=patient_age, visit=visit_type, transcription=transcription)
    response = model.invoke(prompt)
    return response.content
