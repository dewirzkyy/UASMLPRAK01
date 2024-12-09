from flask import Flask, request, render_template
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load dataset
file_path = 'Top-100 Trending Books.csv'
books_data = pd.read_csv(file_path)

# Handle missing values
books_data['genre'] = books_data['genre'].fillna('Unknown')  # Isi genre yang kosong dengan 'Unknown'
books_data['book title'] = books_data['book title'].fillna('Unknown Title')  # Isi judul yang kosong

# Preprocess genre: replace commas with spaces
books_data['genre'] = books_data['genre'].str.replace(',', ' ')

# Vectorize genres using TfidfVectorizer
vectorizer = TfidfVectorizer()  # No tokenizer needed
genre_matrix = vectorizer.fit_transform(books_data['genre'])

# Calculate cosine similarity between books based on genres
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)


def recommend_books(title, books_data, similarity_matrix, top_n=5):
    """
    Recommends books based on a given title using a cosine similarity matrix.
    """
    try:
        # Find the index of the book with the given title
        book_idx = books_data[books_data['book title'] == title].index[0]
    except IndexError:
        return f"Book titled '{title}' not found in the dataset."
    
    # Get similarity scores for the book
    similarity_scores = list(enumerate(similarity_matrix[book_idx]))
    
    # Sort books by similarity scores (excluding itself)
    sorted_books = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    
    # Fetch the titles of the most similar books
    recommended_titles = [books_data.iloc[i[0]]['book title'] for i in sorted_books]
    return recommended_titles


@app.route('/')
def index():
    """
    Render the home page.
    """
    return render_template('index.html')


@app.route('/about')
def about():
    """
    Render the about page.
    """
    return render_template('about.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    """
    Handle form submission and return book recommendations.
    """
    book_title = request.form.get('book_title')  # Get the book title from the form
    recommendations = recommend_books(book_title, books_data, cosine_sim)

    # If recommendations return a message, render error
    if isinstance(recommendations, str):
        return render_template('index.html', error=recommendations)

    return render_template('index.html', recommendations=recommendations, input_title=book_title)


if __name__ == '__main__':
    app.run(debug=True)
