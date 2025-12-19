import flwr.serverapp.strategy as strategy

def get_strategy(context) -> strategy.Strategy:
    # ここでStrategyのカスタマイズを行う
    # デフォルトではFedAvgを使用
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    return strategy.FedAvg(
        fraction_evaluate=fraction_evaluate,
    )