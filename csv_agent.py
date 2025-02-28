from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama
import pandas as pd

# Define the chat model
chat_model = ChatOllama(
    model='llama3.2:latest',
    temperature=0.7,
)

# read csv file
df = pd.read_csv('data/salaries_2023.csv').fillna(value=0)

# print(df.head())

from langchain_experimental.agents.agent_toolkits import (
    create_pandas_dataframe_agent,
    create_csv_agent,
)

# Define the agent
# verbose=True, so that we can see the reasoning from the chat model
agent = create_pandas_dataframe_agent(df=df, llm=chat_model, verbose=True)

# res = agent.invoke("how many rows are there?")

# print(res)

# then let's add some pre and sufix prompt
CSV_PROMPT_PREFIX = """
First set the pandas display options to show all the columns,
get the column names, then answer the question.
"""

CSV_PROMPT_SUFFIX = """
- **ALWAYS** before giving the Final Answer, try another method.
Then reflect on the answers of the two methods you did and ask yourself
if it answers correctly the original question.
If you are not sure, try another method.
FORMAT 4 FIGURES OR MORE WITH COMMAS.
- If the methods tried do not give the same result,reflect and
try again until you have two methods that have the same result.
- If you still cannot arrive to a consistent result, say that
you are not sure of the answer.
- If you are sure of the correct answer, create a beautiful
and thorough response using Markdown.
- **DO NOT MAKE UP AN ANSWER OR USE PRIOR KNOWLEDGE,
ONLY USE THE RESULTS OF THE CALCULATIONS YOU HAVE DONE**.
- **ALWAYS**, as part of your "Final Answer", explain how you got
to the answer on a section that starts with: "\n\nExplanation:\n".
In the explanation, mention the column names that you used to get
to the final answer.
"""

# question = 'Which grade has the highest average base salary, and compare the average female pay vs male pay?'

# res = agent.invoke(CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX)
# print(res['output']) 

import streamlit as st

st.title("Pandas DataFrame Agent")
st.write("This is a simple agent that can answer questions about a pandas dataframe. The agent uses a large language model to reason about the dataframe. You can ask the agent questions about the dataframe, and it will do its best to answer them.")

st.write("### Dataset Preview")
st.write(df.head())

# User input for the question
st.write("### Ask a Question")
question = st.text_input(
    "Enter your question about the dataset:",
    "Which department makes the most on average and give the actual amount?",
)

# Run the agent and display the result
if st.button("Run Query"):
    QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
    res = agent.invoke(QUERY)
    st.write("### Final Answer")
    st.markdown(res['output'])