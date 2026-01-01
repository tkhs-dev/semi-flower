"""pytorchexample: A Flower / PyTorch app."""
import logging
import os

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""

    path = f"results/{context.run_id}"
    os.makedirs(path, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        filename=f"results/{context.run_id}/client_{context.node_config["partition-id"]}.log",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    malicious_count = context.run_config.get("malicious-nodes", 0)

    is_malicious = partition_id < malicious_count
    logging.info(f"Partition ID: {partition_id}, Malicious Nodes: {malicious_count}")
    logging.info("This node acts as malicious node." if is_malicious else "This node acts as benign node.")

    trainloader, _ = load_data(partition_id, num_partitions, batch_size, is_malicious=is_malicious)

    # Call the training function
    train_loss = train_fn(
        model,
        trainloader,
        context.run_config["local-epochs"],
        msg.content["config"]["lr"],
        device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics = {
        "train_loss": train_loss,
        "num-examples": len(trainloader.dataset),
    }
    logging.info(f"Train loss: {train_loss}")
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    logging.basicConfig(
        level=logging.INFO,
        filename=f"results/{context.run_id}/client_{context.node_config["partition-id"]}.log",
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.info(f"Evaluate called on client {context.node_config['partition-id']}")

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    _, valloader = load_data(partition_id, num_partitions, batch_size)

    # Call the evaluation function
    eval_loss, eval_acc = test_fn(
        model,
        valloader,
        device,
    )

    # Construct and return reply Message
    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    logging.info(f"Eval loss: {eval_loss}, Eval accuracy: {eval_acc}")
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)

if __name__ == "__main__":
    app.run()
