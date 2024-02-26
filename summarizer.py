from openai import OpenAI

api_key = "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"

client = OpenAI(api_key=api_key)

def extract_information(user_input):
    context = [
        {"role": "system", "content": "You are a tool for person1 that determines the name and affiliation to person1."},
        {"role": "system", "content": "extract the name and affiliation for the given sentence seperated by a comma."},
        {"role": "system", "content": "If there's no name and affiliation, return Unknown."},
        {"role": "system", "content": "If there's a name but no affiliation, return the name."}
    ]

    # Make a request to the API with context and user input
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        temperature=0.8,
        max_tokens=3000,
        response_format={"type": "text"},
        messages=context + [
            {"role": "user", "content": user_input}
        ]
    )

    output = response.choices[0].message.content

    return output

user_input = "my name is jeff and "
result = extract_information(user_input)
