"""soumya-fed-yolo: A Flower / PyTorch app."""

from flwr.client import NumPyClient, ClientApp
from flwr.common import Context

from soumya_fed_yolo.task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net #! Step 3: Random initialised model passed as variabe to FlowerClient
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

    def fit(self, parameters, config):
        set_weights(self.net, parameters) #! Step 4: Set the weights of the model to the weights received from the server
        results = train(
            self.net, #! Step 5: Train the model on the local data
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
        )
        return get_weights(self.net), len(self.trainloader.dataset), results
        #! Step 6: Return the weights of the model after training on the local data

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net().to(DEVICE) #! Step 1: Random Initialization
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()
    #! Step 2: Random initialised model passed as variabe to FlowerClient


# Flower ClientApp
app = ClientApp(
    client_fn,
)