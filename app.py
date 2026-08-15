import gradio as gr
from sentence_transformers import SentenceTransformer
import torch
from huggingface_hub import InferenceClient

# This is the same pattern from the Generative AI lesson! It uses the
# Inference Provider API to send your messages to an AI model and get
# a response back. Swap out the model below for a different one if
# you want to experiment!
#
# Note: if this Space doesn't already have one, you'll need to add an
# HF_TOKEN secret in the Space's Settings tab for this to work
# (Settings -> Variables and secrets -> New secret).

client = InferenceClient("Qwen/Qwen2.5-7B-Instruct", bill_to="kode-with-klossy")

with open("knowledgebase.txt", "r", encoding="utf-8") as file:
    knowledge_text = file.read()
    
def preprocess_text(text):
    cleaned_text = text.strip()
    chunks = cleaned_text.split("\n")

    cleaned_chunks = []

    for chunk in chunks:
        stripped_chunk = chunk.strip()
        if len(stripped_chunk) > 0:
            cleaned_chunks.append(stripped_chunk)

    return cleaned_chunks

# embed the chunks
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def create_embeddings(text_chunks):
    chunk_embeddings = embedder.encode(text_chunks, convert_to_tensor=True)
    print(chunk_embeddings)
    print(chunk_embeddings.shape)
    return chunk_embeddings


def get_top_chunks(query, chunk_embeddings, text_chunks):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    query_embedding_normalized = query_embedding / query_embedding.norm()
    chunk_embeddings_normalized = chunk_embeddings / chunk_embeddings.norm(dim=1, keepdim=True)
    similarities = torch.matmul(chunk_embeddings_normalized, query_embedding_normalized)
    top_indices = torch.topk(similarities, k=5).indices

    top_chunks = []
    for i in top_indices:
        chunk = text_chunks[i]
        top_chunks.append(chunk)

    return top_chunks

cleaned_chunks = preprocess_text(knowledge_text)
chunk_embeddings = create_embeddings(cleaned_chunks)

def respond(message, history):
    top_results = get_top_chunks(message, chunk_embeddings, cleaned_chunks)
    
    messages = [{"role": "system", "content": 
                 """Your name is KlassicReks. You are an informative classic novel recommender giving book ideas 
                 to curious readers and AP English Literature students. Your primary task
                 is to recommend classic novels from the provided knowledge base based on
                 the user's preferences. 

                 Additionally, use the following emojis in your response as well: 
                 📜🪶☕🕯️🍂⏳🗝️🕰️🎻🎞

                 Recommendation Rules:
                 1. Only recommend books that appear in the provided knowledge base.
                 2. Do not invent or hallucinate book titles, authors, publication dates, characters, etc.
                 3. Prioritize the user's stated preferences when selecting recommendations. Consider:
                     - genre
                     - themes
                     - mood
                     - writing style
                     - time period
                     - country / cultural background
                     - length or reading difficulty
                     - similarity to books or authors the user mentions
                4. Recommend up to three books, ranking them from the best match to the least strong match
                5. Explain why each recommendation matches the user's request.
                6. If the user provides specific preferences, prioritize those preferences.
                7. Do not reveal major plot twists or the ending unless the user specifically asks for spoilers.
                8. If the user's request is vague, you may ask a short follow-up question before recommending any books. 
                9. If the knowledge base does not contain a book that matches the user's request, recommend the closest available option from the knowledge base.
                10. Make sure to distinguish between novels, novellas, short stories, and plays. Do not call a short story a novel. 
                11. Keep the initial recommendations concise. 
                12. If the user asks for recommendations, rank up to three books based on their preferences.
                13. If the user asks for information about a specific book, answer using the retrieved knowledge base rather than generating new recommendation. 
                14. If the user asks for more information about a previously recommended book, provide additional information about that book using the knowledge base.
                15. If the user asks about a book that is not in the knowledge base, explain that it is not currently in the database.

                Response format:
                For each book recommendation, use the following format:

                
                🪶Title of book🪶
                by Author
                
                📜[One sentence description of the premise, 1-2 sentence summary, approximately 30-50 words]

                ☕**Why you'll like it:** [Brief explanation connecting the book the user's preferences.]


                After the recommendations, ask:
                "Would you like a longer summary, themes, historical context, or more books like these?"
                """},
                {"role": "system", "content": f"""Here are the books retrieved from the knowledge base: {top_results}
                
                Use these books as the candidate recommendations. Rank them yourself according to how well they match the user's request. Do not recommend books outside this list."""}]

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": message})

    response = client.chat_completion(
        messages,
        max_tokens=600,
        temperature=0.4
    )
    
    return response.choices[0].message.content.strip()

theme='harsh8001/minimal-orange'

with gr.Blocks() as interface:
    gr.Image(
        value = "background-image.png",
        show_label = False,
        buttons = []
    )
    with gr.Row():
        with gr.Column(scale = 1):
            gr.Markdown("""
            ## KlassicReks
            KlassicReks helps you find classic literature recommendations based on your preferences. 
            Begin your conversation with KlassicReks or select one of the example statements to get started. 🕯️🍂⏳
            """)
        with gr.Column(scale = 2):
            gr.ChatInterface(respond, title = "KlassicReks", examples = ["Recommend a classic for a beginner", "A short classic with a powerful message"])


interface.launch(ssr_mode=False, theme = theme)


# TODO: This is just a starting point! Customize the system prompt,
# the model, and the interface to make this project your own!
