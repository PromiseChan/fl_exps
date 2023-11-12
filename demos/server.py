import flwr as fl


def weighted_average(metrcis):
    accuracies = [num_example * m["accuracy"] for num_example, m in metrcis]
    examples = [num_example for num_example, _ in metrcis]

    avg_accuracy = sum(accuracies) / sum(examples)
    return {"accuracy": avg_accuracy}


fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=fl.server.strategy.FedAvg(
        evaluate_metrics_aggregation_fn=weighted_average
    )
)