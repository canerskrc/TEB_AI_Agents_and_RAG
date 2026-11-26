from langchain.tools import tool
from ollama import chat
from rag import rag_answer
from ml_model import predict_risk


@tool
def general_question_tool(question: str) -> str:
    """Bankacilik mevzuati disindaki genel sorulari cevaplar"""

    response = chat(
        model = "gemma3:4B",
        messages= [
            {
                'role':'user',
                'content':question
            }
        ]
    )
    return response.message.content
@tool
def teb_mevzuat_tool(question: str) -> str:
    """BDDK, MASAK, KVKK, bankacilik mevzuatlari, musteri sirri ve kredi sorularini
    cevaplar."""
    return rag_answer(question)

@tool
def credit_risk_tool(
        income:float,
        debt:float,
        late_payments:int,
):
    """Sentetik verilerle egitilmis ML modeliyle egitim amacli kredi riski tahmini yapar."""
    return predict_risk(income,debt,late_payments)

