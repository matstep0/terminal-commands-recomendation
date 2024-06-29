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

def run_sanity_check(commands_set):

    from engines import TFIDFEngine
    tfidf_engine = TFIDFEngine(commands_set)
    tfidf_engine.fit()

    matches = 0
    for command, description in commands_set.items():
        recommended_command_list = tfidf_engine.recommend_command(description, top_n=3)
        if any(command == recommended_command[0] for recommended_command in recommended_command_list):
            matches += 1
        else:
            print(f"Mismatch: {command} recommended as {recommended_command_list[0][0]}")
            print(f"Description: {command},{commands_set[command]}")
            print(f"Desription of recommended: {recommended_command_list[0][0]},{commands_set[recommended_command_list[0][0]]}")
            print()

    print(f"Total commands: {len(commands_set)}, Matches: {matches}")

def get_user_query():
    query = input("Enter your command query: ")
    return query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool for command recommendation.")
    parser.add_argument('--test', action='store_true', help='Run the sanity check tests')
    parser.add_argument('--top', type=int, help='Number of top recommendations to display')
    args = parser.parse_args()

    dataset_path = './commands_dataset.txt'
    commands_set = load_dataset(dataset_path)

    if args.test:
        run_sanity_check(commands_set)
    else:
        ic("Dataset loaded.")  # You can comment this line out for cleaner output
        
        query = get_user_query()
        ic(f"User query: {query}")  # You can comment this line out for cleaner output

        tfidf_engine = TFIDFEngine(commands_set)
        tfidf_engine.fit()
        
        top_n = 5  # Default value
        if args.top:
            top_n = args.top
        
        recommended_commands = tfidf_engine.recommend_command(query, top_n)
        
        print()
        # Get top n recommendations
        for command, score, description in recommended_commands:
            print(f"Command: {command}, Score: {score}, Description: {description}")