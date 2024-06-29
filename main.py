import os
from icecream import ic
ic.disable()

def load_dataset(file_path):
    commands_set = {}
    with open(file_path, 'r') as file:
        for number, line in enumerate(file):
            command, description = line.strip().split(':', 1)
            commands_set[command] = description
    return commands_set

def get_user_query():
    query = input("Enter your command query: ")
    return query

if __name__ == "__main__":
    dataset_path = './commands_dataset.txt'
    commands_set = load_dataset(dataset_path)
    ic("Dataset loaded.")  # You can comment this line out for cleaner output

    query = get_user_query()
    ic(f"User query: {query}")  # You can comment this line out for cleaner output

    # Initialize and use the engine here
    from engines import TFIDFEngine

    tfidf_engine = TFIDFEngine(commands_set)
    tfidf_engine.fit()
    
    recommended_command = tfidf_engine.recommend_command(query)
    print(f"Recommended command: {recommended_command}")  # You can comment this line out for cleaner output
    
    # Get top 5 recommendations
    top_5_recommendations = tfidf_engine.recommend_top_n(query, 5)
    for command, score in top_5_recommendations:
        print(f"{command}: {score}")
