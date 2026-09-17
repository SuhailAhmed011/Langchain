# Concept

# Semantic splitting divides text based on meaning rather than simply
# counting characters or following document structure.

# It uses embeddings to compare the meaning of neighboring sentences.

# Basic process:

# Document
# ↓
# Sentences
# ↓
# Embeddings
# ↓
# Compare neighboring sentences
# ↓
# Detect semantic change
# ↓
# Create chunks


from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

text = """
Artificial Intelligence is transforming modern businesses.
Machine Learning allows computers to learn patterns from data.
Deep Learning is a subset of Machine Learning.

Football is one of the most popular sports in the world.
Teams compete against each other to score goals.
The FIFA World Cup is one of the biggest football tournaments.
"""

splitter = SemanticChunker(
    embeddings,
    breakpoint_threshold_type="percentile"
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
    print("-" * 50)