# Runnable:
# A common interface in LangChain that allows components to be executed and connected in a standard way.

# Basic idea:
#
# Input → Runnable → Output
#
# Common method:
# .invoke()


# 1. RunnableSequence --------------

# Runs components one after another.
#
# A → B → C
#
# The output of one component becomes the input of the next.

from langchain_core.runnables import RunnableSequence

sequence = RunnableSequence(
    lambda x: x * 2,
    lambda x: x + 10
)

print(sequence.invoke(5))
# Output: 20
#
# 5 → ×2 → 10 → +10 → 20


# Short/modern way using LCEL:
#
# chain = step1 | step2 | step3
#
# Example:
#
# chain = prompt | model | parser


# 2. RunnableParallel -------------------------------------------

# Runs multiple independent tasks at the same time
# using the SAME input.
#
#             ┌→ Task A
# Input ──────┼→ Task B
#             └→ Task C

from langchain_core.runnables import RunnableParallel

parallel = RunnableParallel(
    {
        "double": lambda x: x * 2,
        "square": lambda x: x ** 2
    }
)

print(parallel.invoke(5))

# Output:
# {
#     "double": 10,
#     "square": 25
# }


# 3. RunnablePassthrough ------------------------------------

# Simply passes the input without changing it.
#
# Input → Passthrough → Same Input

from langchain_core.runnables import RunnablePassthrough

passthrough = RunnablePassthrough()

print(passthrough.invoke("Hello"))
# Output: Hello


# Useful when you want to keep the ORIGINAL input
# while also generating another value.


# 4. RunnableLambda ---------------------------------------------

# Converts a normal Python function into a Runnable.
#
# Useful for adding custom Python logic to a chain.

from langchain_core.runnables import RunnableLambda

def word_count(text):
    return len(text.split())

word_count_runnable = RunnableLambda(word_count)

print(word_count_runnable.invoke("LangChain is very useful"))
# Output: 4


# Now the Python function can be used inside a chain:
#
# chain = prompt | model | parser | word_count_runnable


# 5. RunnableBranch  ---------------------------------------------------

# Used for conditional logic.
#
#              Condition
#              /       \
#           True       False
#            ↓           ↓
#         Chain A      Chain B
#
# Similar to:
#
# if condition:
#     A
# else:
#     B

from langchain_core.runnables import RunnableBranch

branch = RunnableBranch(
    (
        lambda x: x > 10,
        lambda x: "Greater than 10"
    ),
    lambda x: "10 or less"
)

print(branch.invoke(15))
# Output: Greater than 10

print(branch.invoke(5))
# Output: 10 or less


# LCEL - LangChain Expression Language -----------------------------------

# The "|" operator is used to compose Runnables.
#
# Example:
#
# prompt | model | parser
#
# Means:
#
# Prompt → Model → Parser
#
# This creates a RunnableSequence.


# QUICK MEMORY. -----------------------------------------------------

# Runnable
#   ↓
# Common interface for LangChain components
#
# Sequence     → A → B → C
# Parallel     → A + B + C
# Passthrough  → Same input
# Lambda       → Python function → Runnable
# Branch       → Condition → A OR B
#
# .invoke()    → Run a Runnable
# "|"          → Connect Runnables



#Execution Methods ------------------------------------

# invoke()  → single input
# batch()   → multiple inputs
# stream()  → output progressively
# ainvoke() → async single input
# abatch()  → async batch
# astream() → async streaming