from langchain_openai import ChatOpenAI
# import os

llm = ChatOpenAI(
    api_key="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"
)

response = llm.invoke("Hello, how are you?")
print(response)

# for chunk in response:
#     print(chunk.content, end="", flush=True)