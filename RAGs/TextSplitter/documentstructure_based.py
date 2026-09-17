# Concept

# Document-structure-based splitters understand the structure of a specific
# document format or programming language.

# Examples:

# Python
# Markdown
# HTML
# JavaScript
# LaTeX

# Instead of treating everything as plain text, the splitter tries to keep
# related structural elements together.


from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language
)

code = """
class Calculator:

    def add(self, a, b):
        return a + b

    def subtract(self, a, b):
        return a - b


def multiply(a, b):
    return a * b
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language = Language.PYTHON,
    chunk_size = 100,
    chunk_overlap = 20
)

chunks = splitter.split_text(code)

for i, chunk in enumerate(chunks):
    print(f"chunk {i + 1}: ")
    print(chunk)
    print("-" * 50)