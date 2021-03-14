import logging

log = logging.getLogger(__name__)

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from src.fed_zoo.base import FedBase
from src.fed_zoo.client import FedAvgClient as Client
from src.fed_zoo.center_server import FedAvgCenterServer as CenterServer

from tqdm import tqdm

from torch.optim.lr_scheduler import StepLR

# 包含了许多ssd的配置，包含类别数量什么的, 比如coco voc
from data import *
from utils.augmentations import SSDAugmentation


class FedAvg(FedBase):
    def __init__(self,
                 model,
                 optimizer,
                 optimizer_args,
                 num_clients=20,
                 local_epoch=1,
                 iid=False,
                 device="cpu",
                 writer=None, args=None):
        self.optimizer = optimizer
        self.optimizer_args = optimizer_args

        self.num_clients = num_clients  # K
        self.local_epoch = local_epoch  # E
        self.args = args

        #  iid to do
        if args.dataset == 'VOC':
            self.cfg = voc
            dataset = VOCDetection(root=args.dataset_root,
                                   transform=SSDAugmentation(self.cfg['min_dim'],
                                                             MEANS))
            test_dataset = VOCDetection(args.dataset_root, [('2007', 'test')], None, VOCAnnotationTransform())
        else:
            self.cfg = coco
            dataset = VOCDetection(root=args.dataset_root,
                                   transform=SSDAugmentation(self.cfg['min_dim'],
                                                             MEANS))
            test_dataset = COCODetection(args.voc_root, [('2007', 'test')], None, VOCAnnotationTransform())

        lengths = [len(dataset) // self.num_clients + (len(dataset) % self.num_clients \
                                                           if i == (self.num_clients - 1) else 0) for i in
                   range(self.num_clients)]

        local_datasets = torch.utils.data.random_split(dataset, lengths)

        local_dataloaders = [
            DataLoader(d,
                       num_workers=args.num_workers,
                       batch_size=args.batch_size,
                       shuffle=True,
                       collate_fn=detection_collate,
                       pin_memory=True)
            for d in local_datasets
        ]

        self.clients = [
            Client(k, local_dataloaders[k], self.cfg, args, device) for k in range(num_clients)
        ]

        self.total_data_size = sum([len(client) for client in self.clients])

        self.aggregation_weights = [
            len(client) / self.total_data_size for client in self.clients
        ]

        test_dataloader = DataLoader(test_dataset,
                                     num_workers=args.num_workers,
                                     batch_size=args.batch_size,
                                     shuffle=True,
                                     collate_fn=detection_collate,
                                     pin_memory=True)

        self.center_server = CenterServer(model, test_dataloader, self.cfg, self.args, device)

        self.loss_fn = CrossEntropyLoss()
        self.writer = writer
        self._round = 0
        self.result = None

    def fit(self, num_round):
        self._round = 0
        self.result = {'loss': [], 'accuracy': []}
        self.validation_step()
        for t in range(num_round):

            print("====>>> round = {}".format(t))
            self._round = t + 1

            self.train_step()
            self.validation_step()

            if t % self.args.save_freq == 0:
                torch.save(self.center_server.send_model().state_dict(),
                           os.path.join(self.args.save_folder, 'ssd-{}.pth'.format(t)))

    def train_step(self):
        self.send_model()
        for k in range(self.num_clients):
            self.clients[k].client_update(self.optimizer, self.optimizer_args,
                                          self.local_epoch, self._round)
        self.center_server.aggregation(self.clients, self.aggregation_weights)

    def send_model(self):
        for client in self.clients:
            client.model = self.center_server.send_model()

    def validation_step(self):
        test_loss, accuracy = self.center_server.validation(self.loss_fn)
        log.info(
            f"[Round: {self._round: 04}] Test set: Average loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%"
        )
        if self.writer is not None:
            self.writer.add_scalar("val/loss", test_loss, self._round)
            self.writer.add_scalar("val/accuracy", accuracy, self._round)

        self.result['loss'].append(test_loss)
        self.result['accuracy'].append(accuracy)
