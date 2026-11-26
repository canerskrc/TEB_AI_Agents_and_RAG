from email import message
from select import select

from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver

from tools import ( general_question_tool, teb_mevzuat_tool,credit_risk_tool)

model = ChatOllama(
    model = 'qwen3:4B',
    temperature=0
)
""""Agent hafizasinda:
- Kullanici mesajlari
- Onceki cevaplar
- Tool cagrilari 
- Tool sonuclari
"""


memory = InMemorySaver()

agent = create_agent(
    model = model,
    tools=[
        general_question_tool,
        teb_mevzuat_tool,
        credit_risk_tool
    ],
    checkpointer=memory,
    system_prompt="""
    Sen arac kullanabilen bir yapay zeka ajanisin. 
    
    Bankacilik mevzuati hakkindaki sorularsa teb_mevzuat_tool kullan.
    Genel bilgi sorularinda general_question_tool kullan. 
    
    Selamlasma ve konusma gecmisiyle ilgili sorularda gereksiz tool cagrisi yapma
    Onceki konusmayi degerlendirebilirsin. 
    
    Devam sorularinda onceki konusmadan yararlanarak tool'a acik ve bagimsiz bir soru gonder.
    Dokumanda bulunmayan mevzuat bilgilerini uretme. 
    TEB'in ic prosedurleri hakkinda varsayim yapma.  
    
    Kredi riski tahmini istenirse credit_risk_tool kullan.
    Aylik gelir, aylik borc odemesi ve gecikme sayisini kullanici mesajindan veya konusma gecmisinden cikar.
    
    Eksik parametre varsa kullaniciya sor. Finansal degerleri kendin uretme
    ML modelinin sonucunu degistirme.
    Kredi onay veya ret karari verme. 

    """
)

config={
    'configurable':{
        'thread_id':'TEB-user-1'
    },
    'recursion_limit':12
}
message_count = 0

while True:
    question = input('\nKullanici: ').strip()

    if question.lower() == 'exit':
        break
    if not question:
        continue
    result = agent.invoke(
        {
            'messages':[
                {
                    'role':'user',
                    'content':question
                }
            ]
        },
        config = config
    )

    new_messages= result['messages'][message_count:]
    message_count = len(result['messages'])
    selected_tools = []

    for message in new_messages:
        for call in getattr(message, 'tool_calls', []):
            selected_tools.append(call['name'])

    print('Kullanilan tool:', selected_tools or 'YOK')
    print('Agent:', result['messages'][-1].content)
