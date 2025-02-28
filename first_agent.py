from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

# Define the chat model
chat_model = ChatOllama(
    model='llama3.2:3b',
    temperature=0.7,
)

messages = [
    SystemMessage(
        content="You are a helpful assistant who is extremely competent as a Computer Scientist! Your name is Rob."
    ),
    HumanMessage("What is a bit, and tell me your name?"),
]

def first_agent(messages):
    responses = chat_model.invoke(messages)
    return responses

def run_agent():
    print("Running the first agent. Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input == "exit":
            print("Goodbye! Thanks for chatting with me.")
            break

        messages = [HumanMessage(content=user_input)]
        responses = first_agent(messages)
        print("Rob:", responses.content)

if __name__ == '__main__':
    run_agent()
