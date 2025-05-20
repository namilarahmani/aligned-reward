from groq import Groq

client = Groq(api_key='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')
completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "user",
            "content": "Our goal is to drive safely and efficiently across a highway. We have two courses of travel with the following values related to car displacement (total distance in meters traveled) and whether a crash occurred:\n\nThe first car moves 5.0 meters closer to its goal and does not crash.\nThe second car moves 2.2974 meters closer to its goal and does not crash.\n\nBased on the information provided, which of the two cars does better for our goal of safe and efficient highway driving?\n\nIf the first car is better, please output 1.\nIf the second car is better, please output 2.\nIf you have no preference, please output -1.\n\nIMPORTANT: Please return your output as just the integer\n"
        }
    ],
    temperature=0.68,
    max_tokens=1024,
    top_p=1,
    stream=True,
    stop=None,
)

for chunk in completion:
    print(chunk.choices[0].delta.content or "", end="")
