PROMPT = """
You are a professional JSON programmer. Your task is to extract data from the text and represent it as JSON.

Output the result ONLY in JSON format.
The JSON structure must be EXACTLY like this:
{
  "entities": [
    {
      "name": "Entity Name",
      "description": "Concise description based ONLY on the text"
    }
    // ... other entities found
  ],
  "relationships": [
    {
	"source": "Source Entity Name", 
	"description": "Relationship Description", 
	"target": "Target Entity Name"
    }
    // ... other relationships found 
  ]
}
**Rules:**
1.  **entities**: List entities with descriptions from the text.
2.  **relationships**: List relationships. Use EXACT names from `entities`.
3.  Use ONLY information from the text.
4.  Your response MUST be ONLY valid JSON. Nothing else. (start with ```json)

**TEXT TO ANALYZE:**
 """
prompt_tempalte = str("""
<|im_start|>system
{system_message}<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
""")

# сторонние импорты
from langchain_openai import ChatOpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
# from set_enviroment import *

from dotenv import load_dotenv
import os

load_dotenv()

# локальные импорты

llm = ChatOpenAI(model = os.getenv("RAG_LLM_MODEL"), base_url = os.getenv('RAG_BASE_URL'), temperature=0.7, top_p=0.9, max_tokens=None, timeout=None) | SimpleJsonOutputParser()


from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_input_files(files_dir):
    'Return list of txt, docx, pdf documents'
    
    def load_file(Loader):
        return Loader(os.path.join(files_dir, file)).load()
    
    files = []
    for file in os.listdir(files_dir):
        if file.endswith(".txt"):
            files += load_file(TextLoader)
        if file.endswith(".docx"):
            files += load_file(Docx2txtLoader)
        if file.endswith(".pdf"):
            files += load_file(PyPDFLoader)

    return files

def get_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Размер чанка
        chunk_overlap=200  # Перекрытие чанков
    )

    return text_splitter.split_documents(documents)

chunks = get_chunks(get_input_files(os.getenv("DATA_PATH")))


responses = [
    llm.invoke(str(prompt_tempalte.format(
        system_message=PROMPT,
        prompt=chunk.page_content
    ))) for chunk in chunks]


results = {
    "entities": [],
    "relationships": []
}

def exctract_data(response):
    # Extract Entities
    for entity in response.get("entities", []):
        # Check if the entity already exists in the results
        name = entity.get("name")
        
        if name is None:
            continue  # Skip if name is not provided

        for existing_entity in results["entities"]:
            if existing_entity["name"] == name:
                description = entity.get("description", False)
                if description:
                    existing_entity["description"].append(description)
                continue

        results["entities"].append({
            "name": entity["name"],
            "description": [entity["description"]]
        })

    # Extract Relationships
    for relationship in response.get("relationships", []):
        # Check if the source and target entities exist in the results
        source = relationship.get("source")
        description = relationship.get("description", False)
        target = relationship.get("target")
        
        if source is None or target is None:
            continue  # Skip if source or target is not provided

        for existing_entity in results["entities"]:
            if existing_entity["name"] == source:
                if description:
                    existing_entity["description"].append(description)
                continue

        results["relationships"].append({
            "source": source,
            "description": description if description else [],
            "target": target
        })

for response in responses:
    # Check if the response is valid JSON
    if isinstance(response, dict):
        exctract_data(response)
    else:
        print("Invalid JSON response:", response)


from json import dump
# Save the results to a JSON file
with open("./output.json", "w") as f:
    dump(results, f, indent=4)

