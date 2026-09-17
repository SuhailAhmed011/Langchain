# Concept

# Text-structure-based splitters use the natural structure of text such as:

# Paragraphs
# New lines
# Spaces
# Characters

# The most commonly used splitter is:

# RecursiveCharacterTextSplitter

# It tries larger separators first and progressively moves to smaller
# separators when necessary.

# Typical hierarchy:

# Paragraph
# ↓
# New Line
# ↓
# Space
# ↓
# Character


from langchain_text_splitters import RecursiveCharacterTextSplitter

text = """
Artificial Intelligence is transforming modern businesses.

Machine Learning allows computers to learn patterns from data.

Generative AI can create text, images, code, and other content.

Large Language Models can understand and generate human-like text.
"""

splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=30
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i + 1}:")
    print(chunk)
    print("-" * 50)