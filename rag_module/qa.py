from saver import init_rag
from minirag import QueryParam

def main():
    # Perform naive search
    # mode="naive"
    # Perform local search
    # mode="local"
    # Perform global search
    # mode="global"
    # Perform hybrid search
    # mode="hybrid"
    # Mix mode Integrates knowledge graph and vector retrieval.
    # mode="mix"

    rag = init_rag()

    print(rag.query(
        "Alya?"
    ))

if __name__ == "__main__":
    main()