import pandas as pd
import os
import json
from ollama import chat
import streamlit as st
from sqlalchemy import create_engine, text
import numpy as np

from langchain.agents import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain.schema import HumanMessage, SystemMessage
from langchain_community.chat_models import ChatOllama

import helpers
from helpers import (
    get_avg_salary_and_female_count_for_division,
    get_total_overtime_pay_for_department,
    get_total_longevity_pay_for_grade,
    get_employee_count_by_gender_in_department,
    get_employees_with_overtime_above,
)

# Define the chat model
chat_model = ChatOllama(
    model='llama3.2:latest',
    temperature=0.7,    
)

# create a db from csv file

# Path to your SQLite database file
database_file_path = "./db/salary.db"

# Create an engine to connect to the SQLite database
# SQLite only requires the path to the database file
engine = create_engine(f"sqlite:///{database_file_path}")
file_url = "./data/salaries_2023.csv"
os.makedirs(os.path.dirname(database_file_path), exist_ok=True)
df = pd.read_csv(file_url).fillna(value=0)
df.to_sql("salaries_2023", con=engine, if_exists="replace", index=False)

db = SQLDatabase.from_uri(f"sqlite:///{database_file_path}")
toolkit = SQLDatabaseToolkit(db=db, llm=chat_model)

def run_conversation(query="""What is the average salary and the count of female employees 
                     in the ABS 85 Administrative Services division?""",):
    messages = [
        # {
        #     "role": "user",
        #     "content": """What is the average salary and the count of female employees
        #               in the ABS 85 Administrative Services division?""",
        # },
        {
            "role": "user",
            "content": query,
        },
        # {
        #     "role": "user", # gives error request too large
        #     "content": """How many employees have overtime pay above 5000?""",
        # },
    ]
    '''
        For openai:
        # Call the model with the conversation and available functions
        response = client.chat.completions.create(
            model=llm_name,
            messages=messages,
            tools=helpers.tools_sql,
            tool_choice="auto",  # auto is default, but we'll be explicit
        )
        response_message = response.choices[0].message
    '''
    response = chat(
        model='llama3.2:latest',
        messages=messages,
        tools=helpers.tools_sql,
    )

    response_text = response.message
    # print(response_text.model_dump_json(indent=2))
    # print(response_text.tool_calls)

    tool_calls = response_text.tool_calls
    if tool_calls:
        available_functions = {
            "get_avg_salary_and_female_count_for_division": get_avg_salary_and_female_count_for_division,
            "get_total_overtime_pay_for_department": get_total_overtime_pay_for_department,
            "get_total_longevity_pay_for_grade": get_total_longevity_pay_for_grade,
            "get_employee_count_by_gender_in_department": get_employee_count_by_gender_in_department,
            "get_employees_with_overtime_above": get_employees_with_overtime_above,
        }
        messages.append(response_text)
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = tool_call.function.arguments
            if function_name == "get_employees_with_overtime_above":
                function_response = function_to_call(amount=function_args.get("amount"))
            elif function_name == "get_total_longevity_pay_for_grade":
                function_response = function_to_call(grade=function_args.get("grade"))
            else:
                function_response = function_to_call(**function_args)
            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                }
            )

            second_response = chat(
                model='llama3.2:latest',
                messages=messages,
            )
    return second_response

if __name__ == "__main__":
    res = (
        run_conversation(
            query="""What is the total longevity pay for employees with the grade 'M3'?"""
        ).message
        .content
    )
    print(res)
    # Step 1: First direct call to the functions =
    # division_name = "ABS 85 Administrative Services"
    # department_name = "Alcohol Beverage Services"
    # grade = "M3"
    # overtime_amount = 5000

    # avg_salary_and_female_count = get_avg_salary_and_female_count_for_division(
    #     division_name
    # )
    # print(
    #     f"Average Salary and Female Count for Division '{division_name}': {avg_salary_and_female_count}"
    # )

