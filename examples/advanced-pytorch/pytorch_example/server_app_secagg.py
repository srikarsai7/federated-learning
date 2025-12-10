"""pytorch-example: Flower + SecAgg+ server app (Fashion-MNIST)."""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from flwr.common import (
    Context,
    Metrics,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server import Grid, LegacyContext, ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.server.workflow.constant import MAIN_PARAMS_RECORD
from torch.utils.data import DataLoader

from pytorch_example.task import (
    Net,
    apply_eval_transforms,
    create_run_dir,
    test,
)


# --------------------------
# Metric Aggregator
# --------------------------
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of client metrics (by number of samples)."""
    if not metrics:
        return {}
    total = sum(num for num, _ in metrics)
    acc = sum(num * m["accuracy"] for num, m in metrics) / total
    return {"accuracy": float(acc)}


# --------------------------
# FedAvg subclass with server-side eval each round
# --------------------------
class LoggingFedAvg(FedAvg):
    """FedAvg that additionally runs centralized evaluation every round.

    After each aggregation, we:
      * rebuild the global model from aggregated parameters
      * evaluate it on the central test set
      * store per-round metrics in `self.server_round_metrics`
    """

    def __init__(self, *, device: str, testloader: DataLoader, **kwargs) -> None:
        super().__init__(**kwargs)
        self.device = device
        self.testloader = testloader
        # List of dicts: {"round": int, "accuracy": float, "loss": float}
        self.server_round_metrics: List[Dict] = []

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ):
        # Let the base FedAvg do the actual aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        # Run centralized evaluation on the aggregated weights
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)

            net = Net()
            state_dict = net.state_dict()
            for (k, _), arr in zip(state_dict.items(), ndarrays):
                state_dict[k] = torch.tensor(arr)
            net.load_state_dict(state_dict)

            loss, acc = test(net, self.testloader, device=self.device)

            self.server_round_metrics.append(
                {
                    "round": int(server_round),
                    "accuracy": float(acc),
                    "loss": float(loss),
                }
            )

        return aggregated_parameters, aggregated_metrics


# --------------------------
# Server App
# --------------------------
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for SecAgg+ server app with per-round logging."""

    # ----------------- Read config -----------------
    cfg = context.run_config
    num_rounds = cfg["num-server-rounds"]
    fraction_train = cfg["fraction-train"]
    fraction_eval = cfg["fraction-evaluate"]
    device = cfg["server-device"]

    # SecAgg+ parameters
    num_shares = cfg["num-shares"]
    reconstruction_threshold = cfg["reconstruction-threshold"]
    max_weight = cfg["max-weight"]

    # ----------------- Output directory -----------------
    save_path, run_dir = create_run_dir(config=cfg)
    print(f"Saving SecAgg+ run outputs under: {save_path} (run: {run_dir})")

    # ----------------- Central test loader (reuse everywhere) -----------------
    global_test_set = load_dataset("zalando-datasets/fashion_mnist")["test"]
    central_testloader = DataLoader(
        global_test_set.with_transform(apply_eval_transforms),
        batch_size=32,
    )

    # ----------------- Model & Initial Parameters -----------------
    net = Net()
    ndarrays = [p.detach().cpu().numpy() for _, p in net.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)

    # ----------------- Strategy (our LoggingFedAvg) -----------------
    strategy = LoggingFedAvg(
        fraction_fit=fraction_train,
        fraction_evaluate=fraction_eval,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        initial_parameters=parameters,
        evaluate_metrics_aggregation_fn=weighted_average,
        device=device,
        testloader=central_testloader,
    )

    # Wrap in LegacyContext for workflows
    legacy_ctx = LegacyContext(
        context=context,
        config=ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # ----------------- SecAgg+ Workflow -----------------
    fit_workflow = SecAggPlusWorkflow(
        num_shares=num_shares,
        reconstruction_threshold=reconstruction_threshold,
        max_weight=max_weight,
    )

    workflow = DefaultWorkflow(fit_workflow=fit_workflow)

    # ----------------- Run Training -----------------
    workflow(grid, legacy_ctx)

    # ----------------- Final Global Parameters -----------------
    final_ndarrays = legacy_ctx.state[MAIN_PARAMS_RECORD].to_numpy_ndarrays()
    final_net = Net()
    state_dict = final_net.state_dict()

    for (k, _), arr in zip(state_dict.items(), final_ndarrays):
        state_dict[k] = torch.tensor(arr)
    final_net.load_state_dict(state_dict)

    # ----------------- Final Global Evaluation -----------------
    final_loss, final_acc = test(final_net, central_testloader, device=device)

    results = {
        "final_global_eval": {
            "accuracy": float(final_acc),
            "loss": float(final_loss),
        },
        "config": cfg,
    }

    results_path = Path(save_path) / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote final metrics to {results_path}")

    # ==========================================================================
    #                           PER-ROUND METRIC LOGGING
    # ==========================================================================

    history = legacy_ctx.history
    round_logs: List[Dict] = []

    # History structure in Flower ≥1.24:
    #   history.losses_distributed  -> List[(round, loss)]
    #   history.metrics_distributed -> Dict[str, List[(round, value)]]
    losses = history.losses_distributed
    rounds = len(losses)

    # Build a map: round -> client-side accuracy
    acc_list = history.metrics_distributed.get("accuracy", [])
    acc_by_round: Dict[int, float] = {rnd: float(val) for rnd, val in acc_list}

    # Build a map: round -> server-side centralized eval from LoggingFedAvg
    server_metrics_by_round: Dict[int, Dict] = {
        m["round"]: m for m in strategy.server_round_metrics
    }

    for idx in range(rounds):
        rnd, loss = losses[idx]  # rnd is the round number Flower recorded
        rnd = int(rnd)

        entry: Dict = {
            "round": rnd,
            "train_metrics": {"train_loss": float(loss)},
        }

        # ----------------- Client-Side Eval (metrics_distributed["accuracy"]) -----------------
        client_acc = acc_by_round.get(rnd)
        if client_acc is not None:
            entry["evaluate_metrics_clientapp"] = {"accuracy": client_acc}
        else:
            entry["evaluate_metrics_clientapp"] = None

        # ----------------- Server-Side Centralized Eval (from LoggingFedAvg) -----------------
        srv = server_metrics_by_round.get(rnd)
        if srv is not None:
            entry["evaluate_metrics_serverapp"] = {
                "accuracy": float(srv["accuracy"]),
                "loss": float(srv["loss"]),
            }
        else:
            entry["evaluate_metrics_serverapp"] = None

        round_logs.append(entry)

    rounds_path = Path(save_path) / "round_metrics.json"
    with open(rounds_path, "w", encoding="utf-8") as f:
        json.dump(round_logs, f, indent=2)

    print(f"Wrote per-round metrics to {rounds_path}")
