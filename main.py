import os
import argparse
from icecream import ic # For debugging 
from engines import TFIDFEngine

def load_dataset(file_path):
    commands_set = {}
    with open(file_path, 'r') as file:
        for number, line in enumerate(file):
            command, description = line.strip().split(':', 1)
            commands_set[command] = description
    return commands_set

def test_model(commands_set,top_n=3):

    from engines import TFIDFEngine
    tfidf_engine = TFIDFEngine(commands_set)
    tfidf_engine.fit()

    matches = 0
    for command, description in commands_set.items():
        recommended_command_list = tfidf_engine.recommend_command(description, top_n=top_n)
        if any(command == recommended_command[0] for recommended_command in recommended_command_list):
            matches += 1
        else:
            print(f"Mismatch: {command} recommended as {recommended_command_list[0][0]}")
            print(f"Description: {command},{commands_set[command]}")
            print(f"Desription of recommended: {recommended_command_list[0][0]},{commands_set[recommended_command_list[0][0]]}")
            print()

    print(f"Total commands: {len(commands_set)}, Matches: {matches}")
    print(f"Top {top_n} accuracy: {matches / len(commands_set) * 100}%")

   

def get_user_query():
    query = input("Enter your command query: ")
    return query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool for command recommendation.")
    parser.add_argument('--test', action='store_true', help='Run the check tests')
    parser.add_argument('--top', type=int, default=5, help='Number of top recommendations to display')
    # Add an argument for specifying a filename with a default value
    parser.add_argument('--filename', type=str, default='./commands_dataset.txt', help='Path to the dataset file')
    args = parser.parse_args()

    # Use the filename specified by the user or the default value
    dataset_path = args.filename
    commands_set = load_dataset(dataset_path)

    if args.test:
        test_model(commands_set=commands_set, top_n=args.top)
    else:
        ic("Dataset loaded.")  # You can comment this line out for cleaner output
        
        query = get_user_query()
        ic(f"User query: {query}")  # You can comment this line out for cleaner output

        tfidf_engine = TFIDFEngine(commands_set)
        tfidf_engine.fit()
        
        top_n = args.top
        
        recommended_commands = tfidf_engine.recommend_command(query, top_n)
        
        print()
        # Get top n recommendations
        for command, score, description in recommended_commands:
            print(f"Command: {command}, Score: {score}, Description: {description}")