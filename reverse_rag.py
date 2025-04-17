from langchain.chat_models import ChatOpenAI
from utils import verify_fact

model = ChatOpenAI(openai_api_key="YOUR_API_KEY", model="gpt-4o")

def extract_facts(soap_notes):
    prompt = f"""Extract individual factual claims from these SOAP notes:

    {soap_notes}

    Return each fact clearly in a numbered list without commentary."""
    response = model.invoke(prompt)
    facts = [fact.strip() for fact in response.content.split('\n') if fact.strip()]
    return facts

def extract_and_verify(soap_notes, transcription):
    facts = extract_facts(soap_notes)
    verified_notes = ""

    for fact in facts:
        similarity, verified = verify_fact(fact, transcription)
        confidence_score = round(similarity, 2)

        if verified:
            verified_notes += f"{fact} ✅ (Confidence: {confidence_score})\n"
        else:
            verified_notes += f"{fact} ❌ (Confidence: {confidence_score} - Verify manually)\n"

    return verified_notes
