from generator import LLMGenerator

llm = LLMGenerator()

context = """
Diabetes is a chronic condition characterized by elevated
blood glucose levels. Common symptoms can include increased
thirst, frequent urination, fatigue and difficulty concentrating.
"""

question = "What are some symptoms of diabetes?"

answer = llm.generate(
    question,
    context
)

print("\nANSWER:")
print(answer)