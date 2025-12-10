# pytorch_example/client_app_secagg.py
"""pytorch-example: Flower + SecAgg+ client app (Fashion-MNIST)."""

from typing import List

import numpy as np
import torch
from flwr.client import ClientApp, NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.common import Context

from pytorch_example.task import Net, load_data
from pytorch_example.task import test as test_fn
from pytorch_example.task import train as train_fn


def get_weights(net: Net) -> List[np.ndarray]:
    """Convert model parameters to list of NumPy arrays."""
    return [p.detach().cpu().numpy() for _, p in net.state_dict().items()]


def set_weights(net: Net, parameters: List[np.ndarray]) -> None:
    """Load list of NumPy arrays into model state_dict."""
    state_dict = net.state_dict()
    new_state = {}
    for (k, _), v in zip(state_dict.items(), parameters):
        new_state[k] = torch.tensor(v)
    net.load_state_dict(new_state, strict=True)


class FlowerClient(NumPyClient):
    """SecAgg+ compatible client using your Fashion-MNIST pipeline."""

    def __init__(
        self,
        partition_id: int,
        num_partitions: int,
        local_epochs: int,
        learning_rate: float,
        device: torch.device,
    ):
        self.net = Net()
        self.device = device
        self.local_epochs = local_epochs
        self.lr = learning_rate

        # Re-use your existing data loader helper
        self.trainloader, self.valloader = load_data(partition_id, num_partitions)

    def fit(self, parameters, config):
        """Train the model on this client's local partition."""
        set_weights(self.net, parameters)

        train_loss = train_fn(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.lr,
            self.device,
        )

        updated_params = get_weights(self.net)
        num_examples = len(self.trainloader.dataset)
        metrics = {"train_loss": float(train_loss)}

        return updated_params, num_examples, metrics

    def evaluate(self, parameters, config):
        """Evaluate the model on this client's local validation split."""
        set_weights(self.net, parameters)

        loss, acc = test_fn(self.net, self.valloader, self.device)
        num_examples = len(self.valloader.dataset)
        metrics = {"accuracy": float(acc)}

        return float(loss), num_examples, metrics


def client_fn(context: Context):
    """Construct a NumPyClient bound to this node's partition."""

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    local_epochs = context.run_config["local-epochs"]
    lr = context.run_config["learning-rate"]

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    return FlowerClient(
        partition_id=partition_id,
        num_partitions=num_partitions,
        local_epochs=local_epochs,
        learning_rate=lr,
        device=device,
    ).to_client()


# Flower ClientApp with SecAgg+ mod enabled
app = ClientApp(
    client_fn=client_fn,
    mods=[secaggplus_mod],
)
