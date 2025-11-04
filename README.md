# Mine Safety AI Agent 🧠

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Enabled-green)](https://www.langchain.com/)

This project is an AI-powered agent designed to analyze Indian mining accident reports. It uses LangChain, LangGraph, and Streamlit to provide a conversational interface for querying structured and unstructured data from the 2015 Directorate General of Mines Safety (DGMS) report.

## 🚀 Problem Statement

Mining accidents are a significant concern in India. Analyzing the vast collections of official accident records from DGMS is a manual, slow, and complex process. This project aims to solve that problem by:
1.  **Digitizing** the data from PDF reports into a queryable format.
2.  **Analyzing** this data to detect hidden patterns, trends, and root causes.
3.  **Providing** an interactive "AI Safety Officer" that allows users to get instant insights and generated reports, ultimately improving safety compliance and hazard identification.

## ✨ Features

* **Interactive Chat Interface:** Ask complex questions about mine safety in plain English.
* **Statistical Analysis:** Get instant answers on accident statistics ("How many people were killed in Orissa from 'Ground Movement'?").
* **Root Cause Analysis:** Find qualitative summaries of *why* accidents happened ("Find me reports where 'unsupported roof' was the cause.").
* **Data Correlation:** Ask questions that combine multiple datasets ("Are dumper accidents more common in iron ore mines?").

## 🛠️ Project Architecture

This project uses a **multi-tool agent** approach. The "AI Brain" (a LangGraph `AgentExecutor`) intelligently routes user questions to one of two specialized tools.



[Image of the AI agent's data-flow architecture]


This architecture is built on two core components:

### 1. The Two Master Datasets

* **Dataset A: `dataset_A_regional_stats.csv` (The "Super-Record")**
    * **What it is:** The "Numbers" dataset. A statistical table with one row per mining region (e.t., Orissa-Iron).
    * **Data:** `total_employees`, `total_killed`, `total_injured`, `hemm_dumper_count`, `explosives_total_kg`.
    * **Source Tables:** 1.2, 2.4, 3.2, 4.6a

* **Dataset B: `dataset_B_accident_case_files.csv` (The "Case Files")**
    * **What it is:** The "Text" dataset. A "case file" for every single fatal accident.
    * **Data:** `date`, `mine_name`, `state`, `mineral`, `summary` (the full text), `findings` (the "why").
    * **Source Tables:** 4.12, 4.6a, 4.7

### 2. The Two AI Tools

* **Tool 1: The Statistical Analyst (Pandas Agent)**
    * **Purpose:** The "Numbers Guy." Answers quantitative questions.
    * **How:** It uses a LangChain agent to run Python Pandas queries on `Dataset A`.

* **Tool 2: The Case File Retriever (RAG Agent)**
    * **Purpose:** The "Reader." Answers qualitative, "why" questions.
    * **How:** It performs semantic search (RAG) over the `summary` and `findings` columns in `Dataset B`.

---

## ⚙️ Data Extraction Pipeline

The two master datasets are created by the `Data_Extraction.ipynb` notebook.

1.  **Load PDF:** The `VOLUME_II_NON_COAL_2015-FULL.pdf` is loaded using `PyMuPDFLoader`.
2.  **Define Pydantic Schemas:** Specific Pydantic models are defined for each table.
3.  **Extract:** LangChain's `with_structured_output` function is called for each table, extracting the data into a structured JSON.
4.  **Merge & Clean:** The extracted JSON files are loaded into Pandas DataFrames.
    * All statistical data (from Tables 1.2, 2.4, 3.2, 4.6a) is cleaned and merged on `state`, `district`, and `mineral` keys to create `dataset_A_regional_stats.csv`.
    * All text-based data (from Table 4.12) is cleaned and saved as `dataset_B_accident_case_files.csv`.

### Pydantic Schemas and Prompts

This section contains all Pydantic models and prompts used in the extraction notebook.

<details>
<summary><b>1. Accident Case Files (Table 4.12)</b></summary>

```python
# Extracting all of the recorded mining accidents
from typing import Literal
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Cause Literals (from Table 4.0) ---
CauseTypeLiteral = Literal[
    "Ground movement", "Transportation machinery (winding)",
    "Transportation machinery (non winding)", "Machinery other than transp. machinery",
    "Explosives", "Electricity", "Dust, gas & other combustible material",
    "Falls (other than fall of ground)", "Other causes"
]

CauseCodeLiteral = Literal[
    "0111", "0112", "0113", "0114", "0115", "0116", "0117", "0118", "0119",
    "0221", "0222", "0223", "0224", "0225", "0228", "0229",
    "0331", "0332", "0333", "0334", "0335", "0336", "0339",
    "0441", "0442", "0443", "0444", "0445", "0446", "0447", "0448", "0449",
    "0551", "0552", "0553", "0554", "0555", "0556", "0557", "0558", "0559",
    "0661", "0662", "0663", "0664", "0665", "0669",
    "0771", "0772", "0774", "0775", "0776", "0777", "0778", "0779",
    "0881", "0882", "0883", "0889",
    "0991", "0992", "0993", "0994", "0995", "0999"
]

CauseDecodeLiteral = Literal[
    "Fall of roof", "Fall of sides (other than overhangs)", "Fall of overhang",
    "Rock burst/bumps", "Air blast", "Premature collapse of workings/pillars",
    "Subsidence", "Landslide", "Collapse of shaft",
    "Overwinding of cages/skip, etc. (upgoing)",
    "Breakage of rope, chain, draw/suspn. gear",
    "Falls of persons from cages, skip, etc.",
    "Falling of objects from cages, skip, etc.", "Hit by cages, skip, etc.",
    "Overwinding of cages/skip (downgoing)", "Other accident due to winding operation",
    "Aerial ropeway", "Rope haulage", "Other rail transportation", "Conveyors",
    "Dumpers", "Wagon movements", "Wheeled trackless (truck, tanker, etc.)",
    "Drilling machines", "Cutting machines", "Loading machines", "Haulage engine",
    "Winding engine", "Shovel, dragline, frontend loader, etc.",
    "Crushing & screening plants", "Other heavy earth moving machinery",
    "Other non-transportation machinery", "Solid blasting projectiles",
    "Deep hole blasting projectiles", "Secondary blasting projectiles",
    "Other projectiles", "Misfires/sockets (while drilling into)",
    "Misfire/socket (other than drilling into)", "Delayed ignition",
    "Blown through shots", "Other explosive accident", "Overhead lines",
    "Trailing cables", "Switch gears, gate end boxes, pommel, etc.",
    "Energized machines", "Power cables other than trailing cables",
    "Other electrical accidents", "Occurrence of gas", "Influx of gas",
    "Explosion/ignition of gas/dust, etc.", "Outbreak of fire or spontaneous heating",
    "Well blowout (with fire)", "Well blowout (without fire)",
    "Other combustible material", "Other accidents due to dust/gas/fire",
    "Fall of person from height/into depth", "Fall of persons on the same level",
    "Fall of objects incl. rolling objects", "Other accident due to falls",
    "Irruption of water", "Flying pieces (except due to explosives)",
    "Drowning in water", "Buried in sands, etc.",
    "Bursting/leakage of oil pipe lines", "Unclassified"
]

class Victim(BaseModel):
  name: str = Field(description="Name of the victim")
  age: int = Field(description="Age of the victim")
  gender: str = Field(description="Gender of the victim")
  occupation: str = Field(description="Occupation of the victim")
  status: str = Field(description="Status of the victim , 'Dead' or 'Alive'")

class MineAccidents(BaseModel):
  date: str = Field(description="Date of the accident")
  time: str = Field(description="Time of the accident")
  mine_name: str = Field(description="Name of the mine where the accident occured")
  owner: str = Field(description="Owner of the mine")
  cause_type : CauseTypeLiteral = Field(description="Type of the cause")
  cause_code : CauseCodeLiteral = Field(description="Code of the cause")
  cause_description : CauseDecodeLiteral = Field(description="Description of the cause")
  state: str = Field(description="State of the mine")
  district: str = Field(description="District of the mine")
  mineral: str = Field(description="Mineral mined")
  victims: list[Victim] = Field(description="List of all victims")
  summary: str = Field(description="Summary of the accident")
  remarks: str = Field(description="Remarks about the accident")

class RecordedAccidents(BaseModel):
  accidents: list[MineAccidents] = Field(description="List of all recorded accidents")

prompt_recorded_accidents = """
Study each of the following accident and list down the details based on the content provided by the user.
"""
<details>
<summary><b>2. Accident Stats by Location (Table 4.6a)</b></summary>

```python
from pydantic import BaseModel, Field
from typing import List

class CasualtyCount(BaseModel):
    """Holds the male and female casualty counts for one location."""
    male: int = Field(0, description="Male casualties")
    female: int = Field(0, description="Female casualties")

class CasualtyDetails(BaseModel):
    """
    A nested dictionary breaking down casualties by the
    place they occurred (Below, Opencast, Above, Total).
    """
    below_ground: CasualtyCount = Field(
        description="Casualties occurring Below Ground"
    )
    opencast_ground: CasualtyCount = Field(
        description="Casualties occurring in Opencast Ground"
    )
    above_ground: CasualtyCount = Field(
        description="Casualties occurring Above Ground"
    )
    total: CasualtyCount = Field(
        description="Total casualties (Male, Female)"
    )

class LocationAccidentStats(BaseModel):
    """
    Represents a single row from Table 4.6a, detailing accidents
    for a specific entity (e.g., a district, a state total).
    """
    entity_name: str = Field(
        description="The name of the mineral, state, or district for the row (e.g., 'EAST GODAVARI', 'TOTAL : ASSAM')"
    )
    fatal_accidents: int = Field(
        description="Number of fatal accidents for this entity"
    )
    serious_accidents: int = Field(
        description="Number of serious accidents for this entity"
    )
    persons_killed: CasualtyDetails = Field(
        description="A nested dictionary of all persons killed, broken down by location"
    )
    persons_seriously_injured: CasualtyDetails = Field(
        description="A nested dictionary of all persons seriously injured, broken down by location"
    )

class AccidentByLocationReport(BaseModel):
    """
    The top-level model to hold the list of all extracted records
    from Table 4.6a. The final output will have a 'locations' key.
    """
    locations: List[LocationAccidentStats] = Field(
        description="A list of all accident records, one for each entity row in the table"
    )

acc_loc_prompt = """
You are an expert data extraction assistant. Your task is to extract all accident statistics from the provided text, which is from Table 4.6a.

The final output must be a JSON object with a single key: "locations". This key must contain a list of all the rows you find.

For *each row* in the table (e.g., "EAST GODAVARI", "TOTAL : ASSAM"), create one object in the "locations" list with the following fields:

1.  `entity_name`: The name of the row (e.g., "EAST GODAVARI", "ALL INDIA : OIL").
2.  `fatal_accidents`: The number under "Number of Accidents -> Fatal".
3.  `serious_accidents`: The number under "Number of Accidents -> Serious".
4.  `persons_killed`: A nested dictionary for the "Number of Persons Killed" section.
5.  `persons_seriously_injured`: A nested dictionary for the "Number of Persons Seriously Injured" section.

For both `persons_killed` and `persons_seriously_injured`, the nested dictionary *must* have this exact structure:
* `below_ground`: A dictionary with `male` and `female` counts.
* `opencast_ground`: A dictionary with `male` and `female` counts.
* `above_ground`: A dictionary with `male` and `female` counts.
* `total`: A dictionary with `male` and `female` counts.

If a value is missing or is a dash, use 0. Process every entity row you see in the table.
"""
```
</details>

<details> <summary><b>5. Machinery (HEMM) (Table 2.4)</b></summary>

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class HemmStats(BaseModel):
    """A nested dictionary holding the counts and HP for one type of HEMM."""
    count: int = Field(0, description="Number of machines of this type (NO.)")
    hp: int = Field(0, description="Horsepower of the machines (H.P.)")

class HemmByMineral(BaseModel):
    """
    Represents one row from Table 2.4, detailing all HEMM
    for a specific mineral in a specific state.
    """
    mineral: str = Field(description="The mineral type (e.g., 'BAUXITE')")
    state: str = Field(description="The state (e.g., 'CHHATTISGARH')")
    
    mines_using_hemm: int = Field(
        description="No. OF MINES USING HEMM"
    )
    
    # Each field below is a nested dictionary
    electrical_shovel: HemmStats
    diesel_shovel: HemmStats
    dumpers: HemmStats
    dozers: HemmStats
    loaders: HemmStats
    
    # These are from the second part of the table
    tractor: Optional[HemmStats] = None
    drag_line: Optional[HemmStats] = None
    grader: Optional[HemmStats] = None
    others: Optional[HemmStats] = None
    total_hemm: Optional[HemmStats] = None

class HemmReport(BaseModel):
    """The main model to hold a list of all HEMM entries."""
    hemm_entries: List[HemmByMineral]

hemm_prompt = """
You are an expert data extraction assistant. Your task is to extract all HEMM (Heavy Earth Moving Machinery) statistics from Table 2.4 (pages 70-79).

The final output must be a JSON object with a single key: "hemm_entries". This key must contain a list of all the rows you find.

For *each row* (e.g., 'BAUXITE' in 'CHHATTISGARH'), create one object in the "hemm_entries" list.

For each object, extract:
1.  `mineral`: The mineral name.
2.  `state`: The state name.
3.  `mines_using_hemm`: The count from 'NO. OF MINES USING HEMM'.

CRITICAL: The following fields MUST be nested dictionaries, each with a 'count' (from NO.) and 'hp' (from H.P.) key:
* `electrical_shovel`
* `diesel_shovel`
* `dumpers`
* `dozers`
* `loaders`
* `tractor` (from the second part of the table)
* `drag_line` (from the second part of the table)
* `grader` (from the second part of the table)
* `others` (from the second part of the table)
* `total_hemm` (from the second part of the table)

If a value is missing or is a dash, use 0.
"""

```

</details>

<details> <summary><b>6. Explosives Consumption (Table 3.2)</b></summary>

```python
from pydantic import BaseModel, Field
from typing import List

class ExplosiveDetonators(BaseModel):
    """Number of electric and ordinary detonators."""
    electric: int = Field(0, description="Number of Electric Detonators")
    ordinary: int = Field(0, description="Number of Ordinary Detonators")

class ExplosiveConsumption(BaseModel):
    """
    Represents one row from Table 3.2, detailing explosives
    consumption (in KGs) for a specific mineral in a specific state.
    """
    mineral: str = Field(description="The mineral type (e.g., 'BAUXITE')")
    state: str = Field(description="The state (e.g., 'CHHATTISGARH')")
    
    mines_using_explosives: int = Field(
        description="No. of Mines Using Explosives"
    )
    
    # All explosive amounts are in KILOGRAMS
    ng_based: int = Field(0, description="N. G. Based")
    an_based: int = Field(0, description="A. N. Based")
    slurries_large_dia: int = Field(0, description="Slurries Large Diameter")
    slurries_small_dia: int = Field(0, description="Slurries Small Diameter")
    boosters: int = Field(0)
    gun_powder: int = Field(0)
    total_explosives_kg: int = Field(description="Total amount of explosives used (in KILOGRAMS)")
    
    detonators: ExplosiveDetonators = Field(
        description="A nested dictionary of detonator counts."
    )

class ExplosivesReport(BaseModel):
    """The main model to hold a list of all explosives entries."""
    explosives_entries: List[ExplosiveConsumption]

explosives_prompt = """
You are an expert data extraction assistant. Your task is to extract all explosives consumption statistics from Table 3.2 (pages 84-89).

The final output must be a JSON object with a single key: "explosives_entries". This key must contain a list of all the rows you find.

For *each row* (e.g., 'BAUXITE' in 'CHHATTISGARH'), create one object in the "explosives_entries" list.

For each object, extract:
1.  `mineral`: The mineral name.
2.  `state`: The state name.
3.  `mines_using_explosives`: The count from 'No. of Mines Using Explosives'.
4.  `ng_based`: KGs from 'N. G. Based'.
5.  `an_based`: KGs from 'A. N. Based'.
6.  `slurries_large_dia`: KGs from 'Slurries Large Diameter'.
7.  `slurries_small_dia`: KGs from 'Slurries Small Diameter'.
8.  `boosters`: KGs from 'Boosters'.
9.  `gun_powder`: KGs from 'Gun Powder'.
10. `total_explosives_kg`: KGs from 'Total amount of explosives used'.

CRITICAL: The 'detonators' field MUST be a nested dictionary:
* `detonators`:
    * `electric`: The number from 'Detonators (In Numbers) -> Electric'.
    * `ordinary`: The number from 'Detonators (In Numbers) -> Ordinary'.

If a number is missing or is a dash, use 0.
"""
```
</details>


##🖥️ AI Agent (Streamlit App)
This is the code for the main mine_safety_app.py file. It loads the CSVs created by the notebook, builds the agent, and launches the chat interface.

<details> <summary><b>mine_safety_app.py</b></summary>

```python
import streamlit as st
import pandas as pd
import os

# Core LangChain components
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import create_pandas_dataframe_agent

# --- 1. SETUP: Create Mock Data (So the app can run) ---
# In your real project, you would skip this and just load your CSVs.

# Dataset A: The "Regional Stats" Table
data_a = {
    "state": ["jharkhand", "rajasthan", "orissa", "karnataka"],
    "district": ["west singhbhum", "jhunjhunu", "keonjhar", "bellary"],
    "mineral": ["iron", "copper", "iron", "iron"],
    "total_employees": [7771, 1178, 13985, 5395],
    "total_killed": [1, 1, 2, 1],
    "total_injured": [1, 2, 1, 1],
    "hemm_dumper_count": [75, 2, 740, 802],
    "explosives_total_kg": [3453831, 669955, 4010572, 404840]
}

# Dataset B: The "Accident Case Files" Table
data_b = {
    "accident_id": ["2015-001", "2015-002", "2015-003", "2015-004"],
    "date": ["2015-05-16", "2015-01-14", "2015-11-08", "2015-02-15"],
    "mine_name": ["KHETRI COPPER MINE", "CHIKLA MANGANESE MINE", "TANTRA-RAIKELA IRON MINE", "VEERBHADRA GRANITE MINE"],
    "state": ["rajasthan", "maharashtra", "orissa", "andhra pradesh"],
    "district": ["jhunjhunu", "bhandara", "sundergarh", "prakasham"],
    "mineral": ["copper", "manganese", "iron", "granite"],
    "summary": [
        "A driller was hit by a mass of stone that fell from an unsupported roof.",
        "A worker was buried under a rock mass that fell from the hanging wall side.",
        "An excavator operator was buried when the bench side slided and the machine fell.",
        "A dumper operator lost control on a steep haul road, jumped, and received fatal injuries."
    ],
    "findings": [
        "Had the workings been kept secured by rock bolts, this could have been averted.",
        "Had the hangwall side been made and kept secured, this could have been averted.",
        "Had the sides of the bench been secured, this accident could have been averted.",
        "Had the gradient of the haul road been maintained properly, this could have been averted."
    ]
}

# Create the CSV files
df_a = pd.DataFrame(data_a)
df_b = pd.DataFrame(data_b)
df_a.to_csv("dataset_A_regional_stats.csv", index=False)
df_b.to_csv("dataset_B_accident_case_files.csv", index=False)

# --- 2. LANGCHAIN: Load Data and Initialize LLM ---

# Load your two master datasets
try:
    df_stats = pd.read_csv("dataset_A_regional_stats.csv")
    df_cases = pd.read_csv("dataset_B_accident_case_files.csv")
except FileNotFoundError:
    st.error("Mock CSV files not found. Please re-run the script.")
    st.stop()

# Initialize the LLM (Gemini)
# This will read the API key from Streamlit's secrets
try:
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0, google_api_key=st.secrets["GOOGLE_API_KEY"])
except KeyError:
    st.error("GOOGLE_API_KEY not found in Streamlit secrets. Please create .streamlit/secrets.toml")
    st.stop()


# --- 3. LANGCHAIN: Create the "AI Brain" Tools ---

# Tool 1: The "Numbers Guy" (Pandas DataFrame Agent)
# This agent can answer statistical questions about Dataset A.
pandas_agent = create_pandas_dataframe_agent(
    llm,
    df_stats,
    agent_type="openai-tools",
    verbose=True
)

# Tool 2: The "Reader" (RAG Agent for Case Files)
# This tool can find relevant accident summaries from Dataset B.
@st.cache_resource
def get_retriever():
    # Create documents from the summaries
    documents = [
        f"Summary: {row['summary']}\nFindings: {row['findings']}"
        for index, row in df_cases.iterrows()
    ]
    
    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=st.secrets["GOOGLE_API_KEY"])
    
    # Create and return a FAISS vector store and retriever
    vector_store = FAISS.from_texts(documents, embedding=embeddings)
    return vector_store.as_retriever()

retriever = get_retriever()

# This is the function the agent will call
def find_relevant_cases(query: str) -> str:
    """
    Finds accident case file summaries relevant to a user's query.
    Returns the summaries as a single string.
    """
    docs = retriever.invoke(query)
    return "\n---\n".join([doc.page_content for doc in docs])

# --- 4. LANGGRAPH: Define the Agent and Tools ---

# Create a list of the tools the agent can use
tools = [
    Tool(
        name="statistical_analyst",
        func=pandas_agent.invoke,
        description="""
        Use this tool to answer quantitative or statistical questions.
        It can find totals, averages, counts, and comparisons from the 'regional_stats' data.
        Example questions:
        'How many people were killed in total?'
        'Which state has the most dumpers?'
        'Compare the number of employees in Jharkhand and Orissa.'
        """
    ),
    Tool(
        name="case_file_retriever",
        func=find_relevant_cases,
        description="""
        Use this tool to find qualitative information about *how* or *why* accidents happened.
        It searches through individual accident summaries and findings.
        Example questions:
        'What are the common causes of dumper accidents?'
        'Find me reports about 'fall of roof'.'
        'Why did the accident at Khetri Copper Mine happen?'
        """
    )
]

# Create the Agent's "Brain" (The Prompt)
agent_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a world-class mine safety analyst.
            Your job is to answer the user's questions about mining safety using the provided tools.

            You have two tools:
            1.  `statistical_analyst`: Use this for any questions about numbers, totals, counts, or comparisons.
            2.  `case_file_retriever`: Use this for any questions about *why* or *how* an accident happened, or to find specific accident reports.

            - If the user asks a statistical question, use `statistical_analyst`.
            - If the user asks for a summary, reason, or specific case, use `case_file_retriever`.
            - For complex questions, you can use multiple tools.
            - Provide a final, clear, and helpful answer based on the tool's output.
            """,
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"), # This is where the agent "thinks"
    ]
)

# Create the agent
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# Create the Agent Executor (This runs the LangGraph)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


# --- 5. STREAMLIT: Build the Chat UI ---

st.title("AI Mine Safety Brain 🧠")
st.caption("This agent uses two tools: a statistical agent (for Dataset A) and a RAG agent (for Dataset B).")

# Set up chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you analyze the 2015 mine safety data?"}
    ]

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about the data..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # Run the agent
            response = agent_executor.invoke({"input": prompt})
            
            # Display the final output
            st.markdown(response["output"])
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response["output"]})

```
</details>

##🚀 How to Run
###1. Installation
Clone this repository and install the required dependencies.

```bash
git clone https://your-repo-url/mine-safety-ai.git
cd mine-safety-ai
pip install streamlit pandas langchain langchain-google-genai langchain-community faiss-cpu langchain-experimental

```
##2. Set Up Your API Key
###Create a Streamlit secrets file:

1.Create a folder: .streamlit

2.Create a file inside it: secrets.toml

3.Add your Google API key (the one from your notebook):

```bash
# .streamlit/secrets.toml
GOOGLE_API_KEY = "AIzaSy...your...key...here"
```

##3. Generate Your Data
Run your Data_Extraction.ipynb notebook from start to finish. This will call the data_extraction function for all the tables and generate the JSON files.

Then, run the Pandas merging code (from our previous conversation) to produce the two final CSV files:

dataset_A_regional_stats.csv

dataset_B_accident_case_files.csv

(Note: The mine_safety_app.py file includes mock data, so it will run even before you do this step, but it will only have 4 rows of data.)
