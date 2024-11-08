from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.tools import Tool
from tools import (get_current_time, 
                    web_search, 
                    update_db, 
                    retrieve, 
                    end_conversation,
                    generate_image)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_experimental.utilities import PythonREPL
import speech_recognition as sr
import os
import pyt2s
from pyt2s.services import stream_elements
import warnings
import time
from rpaudio import AudioSink
import pyt2s.services

warnings.filterwarnings('ignore')
load_dotenv()
python_repl = PythonREPL()
memory = ChatMessageHistory(session_id="test-session")

def listen():
    print()
    print("listening...")
    with sr.Microphone() as source:
        filename = "RA/input.wav"
        recognizer = sr.Recognizer()
        source.pause_threshold = 3
        audio = recognizer.listen(source, phrase_time_limit = None, timeout = 10)
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        text =  recognizer.recognize_google(audio)
        print("you said:", text)
        return text
    except Exception as e:
        print(e)

def speak_text(text):
    data = stream_elements.requestTTS(text, stream_elements.Voice.Salli.value)

    with open('RA/output.mp3', '+wb') as file:
        file.write(data)

    time.sleep(0.2)
    audio = AudioSink().load_audio("RA/output.mp3")
    audio.play()

    while audio.is_playing:
        time.sleep(0.1)
        pyt2s.services


def human_input(query):
    if type(query) != str:
        return 'Please pass a single string as your query to the user'
    print()
    print(query)
    speak_text(query)
    text = listen()
    return text
     
tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    ),
    Tool(
        name="Search",
        func=web_search,
        description="Useful for getting search results from the web, should be used to get facts and news. Always use precise query's, and try to use keywords to get apt results"
    ),
    Tool(
        name="Human_input",
        func=human_input,
        description="If you don't know something that is personal to the user, check the database for that information using the retrieve tool, if it is not present there, then you can ask the user your question using this tool. This should also be used if asking the user something can help you give a better response. Pass a single query string to this function, and it will return what the user says. Always use this over Final Answer, and let the user decide when to end the conversation",
    ),
    Tool(
        name="python_repl",
        description="A Python shell. Use this to to solve mathematical expressions. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    ),
    Tool(
        name="UpdateDB",
        func=update_db,
        description="If there is any personal information about the user, or related to the user that is provided, which you think might be useful in the future, then it must be added to the database. You should always use the retrieve tool before this, to check if that data is already present in the database. Pass a SINGLE string as text. Always add the answer with it's related keywords (as a sentence) so that it comes up when you query the database later."
    ),
    Tool(
        name="Retriever",
        func = retrieve,
        description= "Use this tool to retrieve any personal information about the user. If the information does not exist in the database then use the human tool to ask the user that information and then use the UpdatdeDB tool to add that information to the database for future use, if it is relevant. Pass a string to this tool"
    ),
    Tool(
        name="Exit",
        func = end_conversation,
        description = "Use this tool, when the user says goodbye, or any other parting words. This will end the conversation"
    ),
    Tool(
        name="Generate_image",
        func = generate_image,
        description = "Use this tool to generate an image from text, even though this tool isn't going to return anything to you, the image would directly be shown to the user and saved to the working directory. Pass a sentence separated by '-' eg. A-boy-playing. Do not ask the user to give you a sentence in this format, but format it yourself", 
    )
]

llm = ChatGroq(groq_api_key = os.getenv("GROQ_API_KEY"), model_name = "llama3-70b-8192")

template = """
Answer the following questions as best you can, in a conversational manner. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do, if there are different tools that you have to use in a sequence to answer the question, think about that now.
Action: the action to take, should be one of [{tool_names}], use Human_input to just respond to the user and await for new command. Let the user decide when to finish the conversation. Also, never respond with phrases like (waiting for user's response) etc at the end of your sentences.
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation should be continuously repeated. Note that the original question should NOT be repeated)

Question: {input}
Thought:{agent_scratchpad}
"""

prompt = ChatPromptTemplate.from_template(template)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt,
    stop_sequence=True,
)

def handle_errors(e):
    try:
        pass
    except:
        pass

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=False,
    handle_parsing_errors= handle_errors
)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def main():
    while True:
        try:
            text = listen()
            if text:
                response = agent_with_chat_history.invoke(
                            {"input": text},
                            config={"configurable": {"session_id": "<foo>"}},
                            )
                speak_text(response)
        except Exception as e:
            print("An error occurred:",e)
            return

if __name__ == "__main__":
    main()
