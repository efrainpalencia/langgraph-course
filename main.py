from dotenv import load_dotenv

from graph.graph import app

load_dotenv()

if __name__ == "__main__":
    print("Advanced RAG")

    # Invokes the graph orchestration with a question
    print(app.invoke(
        input={"question": "What is short-term memory?"}))
