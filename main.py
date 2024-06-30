import os
import argparse
from icecream import ic # For debugging 
from engines import TFIDFEngine
from engines import TFIDFEngine

def load_dataset(file_path):
    commands_set = {}
    with open(file_path, 'r') as file:
        for _ , line in enumerate(file):
            if line.startswith('#'):
                continue
            command, description = line.strip().split(':', 1)
            commands_set[command] = description
    return commands_set

def test_model(training_set, test_set ,top_n=3, use_leammatization=False, use_stemming=False, metric='sum'):

    tfidf_engine = TFIDFEngine(training_set, use_lemmatization=use_leammatization, use_stemming=use_stemming)
    tfidf_engine.fit()

    matches = 0
    total_commands = len(test_set)
    for i, (command, description) in enumerate(test_set.items()):
        recommended_command_list = tfidf_engine.recommend_command(description, top_n=top_n, metric=metric)
        if any(command == recommended_command[0] for recommended_command in recommended_command_list):
            matches += 1
        else:
            continue
            """print(f"Mismatch: {command} recommended as {recommended_command_list[0][0]}")
            print(f"Description: {command},{test_set[command]}")
            print(f"Desription of recommended: {recommended_command_list[0][0]},{training_set[recommended_command_list[0][0]]}")
            print()"""

        progress = (i + 1) / total_commands * 100
        print(f"\rProgress: {progress:.2f}%", end='')
    print()
    print(f"Total commands: {total_commands}, Matches: {matches}")
    print(f"Top {top_n} accuracy: {matches / total_commands * 100}%")

   

def get_user_query():
    query = input("Enter your command query: ")
    return query

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI tool for command recommendation.")
    parser.add_argument('--test', action='store_true', help='Run the check tests')
    parser.add_argument('--top', type=int, default=5, help='Number of top recommendations to display')
    # Add an argument for specifying a filename with a default value
    parser.add_argument('--train_filename', type=str, default='./data/commands_dataset.txt', help='Path to the dataset file')
    parser.add_argument('--test_filename', type=str, default='./data/commands_dataset.txt', help='Path to the test dataset file')
    parser.add_argument('--query', type=str, help='Query to recommend commands')
    parser.add_argument('--use_lemmatization', action='store_true', help='Use lemmatization')
    parser.add_argument('--use_stemming', action='store_true', help='Use stemming')
    parser.add_argument('--metric', type=str, default='sum', help='Metric to use for ranking commands')
    args = parser.parse_args()

    # Use the filename specified by the user or the default value
    training_set = load_dataset(args.train_filename)
    test_set = load_dataset(args.test_filename)

    if args.test:
        test_model(training_set=training_set,
                    test_set=test_set,
                    top_n=args.top, 
                    use_leammatization=args.use_lemmatization, 
                    use_stemming=args.use_stemming, 
                    metric=args.metric)
    else:
        if args.query:
            query = args.query
        else:
            query = get_user_query()
        ic(f"User query: {query}")  # You can comment this line out for cleaner output

        tfidf_engine = TFIDFEngine(training_set, top_n=args.top, use_lemmatization=args.use_lemmatization, use_stemming=args.use_stemming)
        tfidf_engine.fit()
        
        top_n = args.top
        
        recommended_commands = tfidf_engine.recommend_command(query, top_n, metric=args.metric)
        
        print()
        # Get top n recommendations
        for command, score, description in recommended_commands:
            print(f"Command: {command}, Score: {score}, Description: {description}")