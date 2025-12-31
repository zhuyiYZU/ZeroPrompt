import openai
import csv

# OpenAI API setup (Replace 'your-openai-api-key' with your actual OpenAI key)
openai.api_key = 'your-openai-api-key'


# Function to generate data using a model
def generate_data(model, prompt, num_entries):
    generated_data = []
    for _ in range(num_entries):
        response = openai.Completion.create(
            model=model,
            prompt=prompt,
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.7
        )
        generated_text = response.choices[0].text.strip()
        generated_data.append(generated_text)
    return generated_data


# Function to create CSV for a category
def create_csv(dataset_name, categories, model, num_entries):
    rows = []
    for idx, category in enumerate(categories):
        label = idx + 1  # Labels start from 1
        # Create a more specific and detailed prompt for each category
        prompt = f"Generate a brief and informative news headline or title related to {category}. Ensure it is concise, clear, and relevant to current trends or discussions in that field."
        generated_data = generate_data(model, prompt, num_entries)

        for text in generated_data:
            rows.append([label, text])

    # Write data to CSV
    filename = f"{dataset_name}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Label", "Text"])
        writer.writerows(rows)
    print(f"CSV for {dataset_name} saved as {filename}")


# Combined categories for all datasets
categories = [
    'politics', 'sports', 'business', 'technology',
    'entertainment', 'health', 'us', 'world',
    'computers', 'culture-arts-entertainment', 'education-science',
    'engineering', 'politics-society', 'positive', 'negative'
]

# Model names (replace these with actual available models for each dataset)
models = {
    'general': 'gpt-4',  # GPT-4 for all categories
}

# Generate data for the combined dataset
create_csv('combined_dataset', categories, models['general'], 50)
