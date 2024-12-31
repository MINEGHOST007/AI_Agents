import typer
from typing import Optional, List
from phi.assistant import Assistant
from phi.storage.assistant.postgres import PgAssistantStorage
from phi.knowledge.pdf import PDFUrlKnowledgeBase
from phi.vectordb.pgvector import PgVector2
import os
from dotenv import load_dotenv
import groq

# Load environment variables
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq.api_key = os.getenv("GROQ_API_KEY")

# Database connection string
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

# Knowledge base setup with Groq
knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url)
)

# Load the knowledge base
knowledge_base.load()

# Storage setup
storage = PgAssistantStorage(table_name="pdf_assistant", db_url=db_url)

# Define the PDF Assistant function
def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    # Check for existing runs
    if not new:
        existing_run_ids: List[str] = storage.get_all_run_ids(user)
        if len(existing_run_ids) > 0:
            run_id = existing_run_ids[0]
    
    # Create assistant instance
    assistant = Assistant(
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        show_tool_calls=True,
        search_knowledge=True,
        read_chat_history=True,
    )

    # Print run information
    if run_id is None:
        run_id = assistant.run_id
        print(f"Assistant run_id: {run_id}\n")
    else:
        print(f"Assistant run_id: {run_id} (existing run)\n")
    
    # Start CLI app
    assistant.cli_app(markdown=True)

# Entry point for the script
if __name__ == "__main__":
    typer.run(pdf_assistant)
