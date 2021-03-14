import copy
from collections import OrderedDict

import torch


class CenterServer:
    def __init__(self, model, dataloader, cfg, args, device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.cfg = cfg
        self.args = args
        self.device = device

    def aggregation(self):
        raise NotImplementedError

    def send_model(self):
        return copy.deepcopy(self.model)

    def validation(self):
        raise NotImplementedError


class FedAvgCenterServer(CenterServer):
    def __init__(self, model, dataloader, cfg, args, device="cpu"):
        super().__init__(model, dataloader, cfg, args, device)

    def aggregation(self, clients, aggregation_weights):
        print("Center Server Aggregation ...")
        update_state = OrderedDict()

        for k, client in enumerate(clients):
            local_state = client.model.state_dict()
            for key in self.model.state_dict().keys():
                if k == 0:
                    update_state[
                        key] = local_state[key] * aggregation_weights[k]
                else:
                    update_state[
                        key] += local_state[key] * aggregation_weights[k]

        self.model.load_state_dict(update_state)

    # to do
    def validation(self, loss_fn):
        test_loss = 0
        correct = 0

        # to do
        print("center server validation to do ")

        # self.model.to("cpu")
        test_loss = test_loss / len(self.dataloader)
        accuracy = 100. * correct / len(self.dataloader.dataset)

        return test_loss, accuracy
