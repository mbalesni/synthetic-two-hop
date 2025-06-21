import time
from together import Together

# Initialize the client
client = Together()

years = [
    # Validated ones
    1903,  # Wright brothers
    1912,  # Titanic
    1941,  # Pearl Harbor
    1957,  # Sputnik
    1963,  # Kennedy
    1969,  # Moon Landing
    1986,  # Chernobyl
    1989,  # Berlin Wall
    2005,  # Katrina
    1215,  # Magna Carta signed
    1347,  # Black Death reaches Europe
    1492,  # Columbus reaches Americas
    1776,  # Declaration of Independence
    1789,  # Bastille Day/French Revolution starts ("French Revolution")
    1815,  # Waterloo battle
    1883,  # Krakatoa eruption
    1889,  # Eiffel Tower completed
    1901,  # Queen Victoria dies
    1917,  # Russian Revolution
    1944,  # D-Day invasion
    1950,  # Korean War begins
]

def get_defining_event(year):
    prompt = f"What was the single most significant historical event of {year}? Please respond with just the event name, no explanation."
    
    response = client.chat.completions.create(
        model="meta-llama/Llama-3-8b-chat-hf",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=100,
        temperature=0,  # Set to 0 for deterministic responses
        top_p=1,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"],
    )
    
    # Extract the response content
    if hasattr(response.choices[0], 'message'):
        return response.choices[0].message.content.strip()
    return response.choices[0].delta.content.strip()

def main():
    results = {}
    
    for year in years:
        print(f"\nQuerying for year {year}...")
        try:
            event = get_defining_event(year)
            results[year] = event
            print(f"{year}: {event}")
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"Error for {year}: {e}")
    
    print("\nFinal Results:")
    for year, event in sorted(results.items()):
        print(f"{year}: {event}")

if __name__ == "__main__":
    main()
