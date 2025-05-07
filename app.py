from agent_conditions import agent
# state = {"text": "Как вернуть товар?"}
# state = {"text": "Заказ 563526 пришел поврежденный. Можно оформить возврат?"}
# state = {"text": "не могу отменить заказ"}
# state = {"text": "холодильник отлично работает, быстрая доставка. спасибо!"}

# state = {"text": "Как изменить адрес доставки?"}
# state = {"text": "Какой срок гарантии на заказ 56352?"}
# state = {"text": "йоу"}
state = {"text": "безобразие!!! доставка опоздала на две недели!!!!"}
result = agent.invoke(state)
print(result)
print(result["response"])

# python app.py