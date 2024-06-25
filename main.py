
import streamlit as st
import numpy as np
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from scipy.spatial.distance import cosine as cosine_similarity 

MODEL_NAME = 'textembedding-gecko@003'

def embed_text(texts: list, model_name: str = MODEL_NAME) -> list:
    try:
        model = TextEmbeddingModel.from_pretrained(model_name)
        inputs = [TextEmbeddingInput(text) for text in texts]
        embeddings = model.get_embeddings(inputs)
        return [np.array(embedding.values) for embedding in embeddings]
    except Exception as e:
        raise RuntimeError(f"Error embedding texts: {e}")

def cosine_similarity_percentage(vec1: np.ndarray, vec2: np.ndarray) -> float:
    similarity = 100 * (1 - cosine_similarity(vec1, vec2))
    return similarity 

# Streamlit 
st.title('cosine similaroty')
word1 = st.text_input('first word:')
word2 = st.text_input('second word:')

if st.button('Calculate Similarity'):
    if word1 and word2:
        try:
            embeddings = embed_text([word1, word2], MODEL_NAME)
            similarity_percentage = cosine_similarity_percentage(embeddings[0], embeddings[1])
            st.subheader(f"The similarity between '{word1}' and '{word2}' is: {similarity_percentage:.2f}%")

           
        except RuntimeError as e:
            st.error(f"RuntimeError: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.warning('Please enter both words')
