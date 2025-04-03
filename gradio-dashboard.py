import pandas as pd
import numpy as np
from dotenv import load_dotenv

# Importing required libraries for document loading, embeddings, and vector database
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

import gradio as gr  # Importing Gradio for building the interactive dashboard

# Load environment variables (e.g., OpenAI API key)
load_dotenv()

# Load the book dataset
books = pd.read_csv("books_with_emotions.csv")

# Generate a larger thumbnail version by modifying the URL
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"

# Replace missing thumbnails with a placeholder image
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# Load the book descriptions from a text file
raw_documents = TextLoader("tagged_description.txt").load()

# Split the text into smaller chunks for embedding processing
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=0, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Create a Chroma vector database using OpenAI embeddings
db_books = Chroma.from_documents(documents, OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    """
    Retrieves book recommendations based on semantic similarity search.

    Args:
        query (str): User input text for search.
        category (str, optional): Book category filter.
        tone (str, optional): Emotional tone filter.
        initial_top_k (int, optional): Number of initial retrieved books before filtering.
        final_top_k (int, optional): Final number of books to return after filtering.

    Returns:
        pd.DataFrame: Filtered book recommendations.
    """

    # Perform semantic similarity search on the Chroma database
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Extract book ISBN numbers from search results
    books_list = [int(rec.page_content.strip('"').split()[0]) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(initial_top_k)

    # Apply category filter if selected
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    # Apply emotional tone-based sorting
    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    return book_recs


def recommend_books(query: str, category: str, tone: str):
    """
    Generates a list of book recommendations in a format suitable for Gradio.

    Args:
        query (str): User input query.
        category (str): Selected book category.
        tone (str): Selected emotional tone.

    Returns:
        list: List of tuples containing book image and description.
    """

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"]
        truncated_desc_split = description.split()
        truncated_description = " ".join(truncated_desc_split[:30]) + "..."  # Shorten description

        # Format author names correctly
        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        # Create formatted caption with book title, authors, and description
        caption = f"{row['title']} by {authors_str}: {truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


# Generate category and tone options
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

# Create a Gradio interactive UI for the book recommender
with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# Semantic Book Recommender")  # Title

    with gr.Row():
        # User input for book description
        user_query = gr.Textbox(label="Please enter a description of a book:",
                                placeholder="e.g., A story about forgiveness")
        # Dropdowns for category and tone selection
        category_dropdown = gr.Dropdown(choices=categories, label="Select a category:", value="All")
        tone_dropdown = gr.Dropdown(choices=tones, label="Select an emotional tone:", value="All")
        submit_button = gr.Button("Find recommendations")  # Search button

    gr.Markdown("## Recommendations")  # Section for displaying recommendations
    output = gr.Gallery(label="Recommended books", columns=8, rows=2)  # Display books as a gallery

    # Link the function to UI elements
    submit_button.click(fn=recommend_books,
                        inputs=[user_query, category_dropdown, tone_dropdown],
                        outputs=output)

# Run the application
if __name__ == "__main__":
    dashboard.launch()
