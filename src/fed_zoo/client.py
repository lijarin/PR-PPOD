from torch.autograd import Variable
from layers.modules import MultiBoxLoss
from tqdm import tqdm
from utils.utils import adjust_learning_rate


class Client:
    def __init__(self, client_id, dataloader, cfg, args, device='cpu'):
        self.client_id = client_id
        self.dataloader = dataloader
        self.device = device
        self.__model = None
        self.cfg = cfg
        self.args = args
        self.index_step = 1

    @property
    def model(self):
        return self.__model

    @model.setter
    def model(self, model):
        self.__model = model

    def client_update(self, optimizer, optimizer_args, local_epoch, n_round):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataloader.dataset)


class FedAvgClient(Client):
    def client_update(self, optimizer, optimizer_args, local_epoch, n_round):
        self.model.train()
        self.model.to(self.device)

        optimizer = optimizer(self.model.parameters(), **optimizer_args)
        if n_round in self.args.lr_stage == 0:
            adjust_learning_rate(optimizer, self.args.lr, 0.1, self.index_step)
            self.index_step += 1

        # to do
        criterion = MultiBoxLoss(self.cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, self.args.cuda)

        print("Client training ...")
        for epoch in range(local_epoch):
            for images, targets in tqdm(self.dataloader):
                # load train data
                if self.args.cuda:
                    images = Variable(images.cuda())
                    targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
                else:
                    images = Variable(images)
                    targets = [Variable(ann, volatile=True) for ann in targets]

                # forward
                out = self.model(images)
                # backprop
                optimizer.zero_grad()
                loss_l, loss_c = criterion(out, targets)
                loss = loss_l + loss_c
                loss.backward()
                optimizer.step()

        self.model.to("cpu")
