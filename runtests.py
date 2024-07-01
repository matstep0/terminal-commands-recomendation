from main import test_model, load_dataset

def print_markdown_table(results):
    # Print table header
    print("| Metric | Test 1 | Test 2 | Test 3 |")
    print("|--------|--------|--------|--------|")
    
    # Print table rows
    for metric, accuracies in results.items():
        row = f"| {metric} |"
        for accuracy in accuracies:
            row += f" {accuracy:.2f}% |"
        print(row)

if __name__ == "__main__":
    # Load datasets
    training_set = load_dataset("./data/commands_dataset.txt")
    test1_dataset = load_dataset("./data/test1_dataset.txt")
    test2_dataset = load_dataset("./data/test2_dataset.txt")
    test3_dataset = load_dataset("./data/test3_dataset.txt")
    metrics = ["sum", "cosine", "kld", "pearson", "jsd"]
    results = {metric: [] for metric in metrics}
    # Test each metric
    for metric in metrics:
        for test_set in [test1_dataset, test2_dataset, test3_dataset]:
            accuracy = test_model(training_set, test_set, metric=metric, suppress_output=True, top_n=3, use_lemmatization=False, use_stemming=False)
            results[metric].append(accuracy)
    # Assuming a function to print the markdown table based on results
    print_markdown_table(results)
