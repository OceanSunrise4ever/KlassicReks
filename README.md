KlassicReks

An AI Chatbot that recommends classic books to read based on your input. For people who are looking for book recommendations in Classic Literature or would like to build a reading list for AP English Literature, our KlassicReks ChatBot can provide a variety of classics based on your input (ex. author, length, theme). 

Classic novels have a universal appeal and can connect across multiple generations and centuries. Classics also often relate to unique human experiences and can mean something meaningfully different for everyone.


🤗 **Originally built as a Hugging Face Space:** (https://huggingface.co/spaces/kode-with-klossy/4.4-groupC1-capstone)

> ⚠️ Note: This Space is no longer live. The code in this repo is the full project.

<img width="1360" height="631" alt="image" src="https://github.com/user-attachments/assets/86cbcfe6-745a-4d4d-a5de-23accf24a2cd" />


## What it does

- Custom recommendations based on your interests, including: theme, author, difficulty, time period, etc.
- Obtain detailed book summaries of each book
- Ability to prompt the AI further based on what you need/want
- Enjoy the custom, aesthetic interface


## How it works

When the user first inputs a prompt, the code chunks the text and embeds it so that the AI Model understands and understands the semantic. Using that information, it searches the knowledge base looking for the top 3 books that match your interests.

## Built with

- **Gradio** — the interface
- **Hugging Face Inference Providers** — Qwen/Qwen2.5-7B-Instruct
- **Sentence Transformers** — We used the embedding of text to turn words into numbers. The ChatBot would then use the numbers to find the most relevant context and meaning for user questions
- **Accuracy** — Orders the best matches to improve accuracy.

## What I learned

The most challenging part of building this project was understanding how systems process semantic inputs using vector embeddings to provide relevant context to the LLM. I learned how vector databases calculate the mathematical angles (using cosine similarity, a smaller angle means the texts have highly similar semantic meanings, while a larger angle means they are unrelated.) between embeddings to determine semantic similarity, allowing the chatbot to retrieve highly accurate results based on meaning rather than just keywords. During our test-phase of this chatbot, we learned that our ChatBot was sourcing information outside of the knowledge base and sometimes hallucinated. To fix this problem, we added more rules to the role of the ChatBot to ensure that it was only sourcing form the knowledge base. Another challenge we encountered was importing our custom color palette into Gradio. As this custom palatte was not natively supported by Gradio, it didn't support it when we ran the code. To fix this issue, we chose a similar theme that was supported by Gradio. 

## About

Built at [Kode With Klossy](https://www.kodewithklossy.com) AI/ML Camp,
Summer 2026, by SaanviSri K, Ella W, Aileen L, Rosabella C, Emily X


