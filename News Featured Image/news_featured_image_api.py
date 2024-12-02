from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from docx import Document
import pandas as pd
import numpy as np
import requests
from keybert import KeyBERT

api_key = "AIzaSyBbmOlyteb-YIqfCJd0pLusArzXTPNFW4A"
cx = "03df045159ade4b12"

app = FastAPI()

# Load the pre-trained BERT model
model = KeyBERT('all-MiniLM-L6-v2')

def read_word_file(file_path: str) -> str:
    """
    Reads text from a Word document using python-docx.
    """
    try:
        doc = Document(file_path)
        full_text = []
        for paragraph in doc.paragraphs:
            full_text.append(paragraph.text)
        return "\n".join(full_text)
    except Exception as e:
        raise ValueError(f"Error reading Word document: {e}")

# Function to search Google Custom Search Engine
def google_custom_search(query, api_key, cx, num_results=5):
    search_url = f"https://www.googleapis.com/customsearch/v1"
    
    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": num_results,
        'searchType': "image"
    }

    response = requests.get(search_url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None
  
# Function to extract titles, snippets, and image URLs from search results
def extract_images_and_links(search_results):
    images_and_links = {}
    if "items" in search_results:
        for item in search_results["items"]:
            title = item.get("title", "No Title")
            image = item.get("link", "No Image URL")
            source = item.get("displayLink", "No Image Source")
            link = item.get("image", {}).get("contextLink", "No Content Link")
            images_and_links[title] = {"image": image, "content_link": link, "source": source}
    return images_and_links

@app.post("/upload/")
async def upload_word_file(file: UploadFile = File(...)):
    """
    Endpoint to upload a Word document and extract its content.
    """
    if file.content_type != "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        raise HTTPException(status_code=400, detail="Invalid file format. Only .docx files are supported.")

    try:
        # Save the uploaded file temporarily
        file_location = f"temp_{file.filename}"
        with open(file_location, "wb") as f:
            f.write(await file.read())

        # Read the Word file content
        content = read_word_file(file_location)

        # Extract keywords
        keywords = model.extract_keywords(content, keyphrase_ngram_range=(1, 2), stop_words='english', top_n=5)
        result = google_custom_search(keywords[1][0], api_key, cx, num_results=5)
        images_link = extract_images_and_links(result)

        # Clean up temporary file
        import os
        os.remove(file_location)

        # Return extracted content
        return JSONResponse(content={"result": images_link})
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

