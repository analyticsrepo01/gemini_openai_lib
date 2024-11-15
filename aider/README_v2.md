# Search Application

This application allows you to search through your internal documents using Vertex AI Search and various Large Language Models (LLMs).  It provides features such as filtering, re-ranking, snippets, extractive answers, and summarization.

## Search.py

This script provides the core search functionality.  It leverages the Google Cloud Discovery Engine API to perform searches against a specified data store.  Key features include:

* **Search with Filters:** Allows users to refine search results using structured filter queries.
* **LLM Integration:** Supports multiple LLMs, including Vertex AI Search (default), Gemini 1.5 Flash, and Gemini 1.5 Pro, to generate answers based on search results.
* **Re-ranking:** Optionally re-ranks search results based on semantic relevance to the search query.
* **Result Customization:**  Control the number of results, snippets, extractive answers, and summary results.
* **UI Integration with Streamlit:** Provides an interactive user interface for inputting search queries and displaying results.

The script uses environment variables for project configuration.

## app.py

This script implements a simple Flask app with a `/hello` endpoint that returns "Hello, World!".

## Usage

1.  **Install required libraries:**

    ```bash
    pip install -r requirements.txt
    pip install Flask
    ```

2.  **Set environment variables (for Search.py):**

    ```bash
    export PROJECT_ID=<your-project-id>
    export LOCATION=<your-location>
    export DATASTORE_ID=<your-datastore-id>
    ```

3.  **Run Search.py:**

    ```bash
    streamlit run Search.py <your-project-id> <your-datastore-id>
    ```

4. **Run app.py:**

    ```bash
    python app.py
    ```
