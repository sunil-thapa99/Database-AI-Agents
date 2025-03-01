from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
import pandas as pd
import os
from sqlalchemy import create_engine

# Define the chat model
chat_model = ChatOllama(
    model='llama3.2:latest',
    temperature=0.7,    
)

from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
 
database_file_path = "db/salary.db"

# Create a db from CSV file
engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "data/salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)

# Define the agent
db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=chat_model)

QUESTION = """How many employees are in the ABS 85 Administrative services
and their avg salaries, and also how many of them are female?"""

agent = create_sql_agent(
    toolkit=toolkit,
    llm=chat_model,
    verbose=True,
)

res = agent.invoke(QUESTION)
print(res)

