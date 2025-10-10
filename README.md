# Geek-Bot Movie & Series Recommendation Chatbot
This project is a part of the Akbank GenAI Bootcamp 2025. It features a chatbot developed using the Retrieval-Augmented Generation (RAG) architecture to provide users with personalized movie and series recommendations.

1. Project Aim
The primary goal of this project is to develop an intelligent chatbot that leverages a Large Language Model (LLM) to offer relevant and conversational recommendations for movies and TV series. By implementing a RAG-based system, the chatbot can provide suggestions based on a comprehensive dataset of film and television information, going beyond pre-programmed responses.

2. Dataset
Dataset Used: IMDb v2 on Kaggle

Description: This dataset contains a wide range of information about movies and TV series, including titles, plot summaries, genres, cast, and ratings. The rich textual data, especially the plot summaries, is ideal for creating vector embeddings. This information serves as the external knowledge base for our RAG model, allowing the chatbot to retrieve relevant context before generating a recommendation.

Of course. Let's prepare the README.md file in English together. Below is a template based on your project description and the requirements from your project file. You can copy and paste this into your README.md file on GitHub and fill in the specific details as you complete each step.

Movie & Series Recommendation Chatbot
This project is a part of the Akbank GenAI Bootcamp. It features a chatbot developed using the Retrieval-Augmented Generation (RAG) architecture to provide users with personalized movie and series recommendations.

1. Project Aim
The primary goal of this project is to develop an intelligent chatbot that leverages a Large Language Model (LLM) to offer relevant and conversational recommendations for movies and TV series. By implementing a RAG-based system, the chatbot can provide suggestions based on a comprehensive dataset of film and television information, going beyond pre-programmed responses.


2. Dataset
Dataset Used: IMDb v2 on Kaggle

Description: This dataset contains a wide range of information about movies and TV series, including titles, plot summaries, genres, cast, and ratings. The rich textual data, especially the plot summaries, is ideal for creating vector embeddings. This information serves as the external knowledge base for our RAG model, allowing the chatbot to retrieve relevant context before generating a recommendation.

3. Methods & Technologies
The solution is built upon a modern stack of AI and development tools:

Generation Model: Llama 3.2 is used as the core language model for generating human-like, context-aware responses.

Architecture: The system is built on a Retrieval-Augmented Generation (RAG) pipeline. This architecture enhances the LLM's responses by first retrieving relevant information from a specialized knowledge base (our movie dataset) and then feeding that information to the model as context.

RAG Framework: LangChain is used to structure and manage the RAG pipeline, connecting the different components like the embedding model, vector database, and the generation model.

Embedding Model: An open-source model from sentence-transformers is used to convert the textual data from the movie dataset into numerical vector representations.

Vector Database: FAISS is employed as the vector database to efficiently store and search through the movie data embeddings, enabling fast retrieval of relevant context for any user query.

Development Environment: The project is developed in Google Colab using a T4 GPU runtime. Version control and code repository are managed through GitHub

4. Installation & Usage Guide
To run this project on your local machine, please follow these steps:

Clone the Repository:

Bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
Create a Virtual Environment:

Bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
Install Dependencies:
A requirements.txt file is included to manage all necessary libraries. Install them using the following command:

Bash
pip install -r requirements.txt

Run the Application:
Execute the main Python script to launch the web interface:

Bash
streamlit run app.py

5. Web Interface & Deployment
The chatbot is accessible through a user-friendly web interface created with Streamlit.


Deployment Link: [You will add your live web application link here] ******

How to Use It:

Navigate to the deployment link provided above.

You will see a chat interface with a text input box.

Type your request in the box. You can ask for recommendations based on genre, plot, actors, or just describe the kind of movie you're in the mood for.

The chatbot will process your request and provide a recommendation.


********(You can add screenshots or a short GIF/video here to demonstrate how the application works) 


6. Results

(This section should be filled out after the project is complete) 

In this section, you will summarize the outcomes of your project. Discuss the chatbot's performance, its ability to understand different types of queries, and the quality of its recommendations. You can also mention any challenges you faced and potential improvements for the future.
