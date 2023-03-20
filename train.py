import torch

from evaluation import Evaluation
class Train():
    def __init__(self,
                 train_loader,
                 model:torch.nn.Module,
                 device:torch.device,
                 criterion,
                 optim:torch.optim,
                 epochs:int,
                 test_loader,
                 top_k:int) -> object:
        self.epochs =epochs
        self.device = device
        self.model = model
        self.criterion = criterion
        self.optimizer = optim
        self.dataloader = train_loader
        self.test_loader = test_loader
        self.top_k = top_k

    def train(self):
        epochs = self.epochs
        model = self.model
        criterion = self.criterion
        optimizer  = self.optimizer
        dataloader = self.dataloader
        device = self.device
        top_k = self.top_k
        for epoch in range(epochs):
            avg_cost = 0
            total_batch = len(dataloader)

            for idx,(users,pos_items,neg_items) in enumerate(dataloader):
                users,pos_items,neg_items = users.to(device),pos_items.to(device),neg_items.to(device) #将数据转移到指定的设备上
                user_embeddings, pos_item_embeddings, neg_item_embeddings= model(users,pos_items,neg_items,use_dropout=True)

                optimizer.zero_grad() #将梯度清零，以避免梯度累积
                cost  = criterion(user_embeddings,pos_item_embeddings,neg_item_embeddings)
                cost.backward() #计算损失关于每个参数的梯度
                optimizer.step() #更新模型的参数，以便在训练过程中逐步优化模型
                avg_cost+=cost #该批次的平均成本（avg_cost）添加到总成本中
            avg_cost = avg_cost/total_batch
            eval = Evaluation(test_dataloader=self.test_loader,
                              model = model,
                              top_k=top_k,
                              device=device)
            HR,NDCG = eval.get_metric()
            print(f'Epoch: {(epoch + 1):04}, {criterion._get_name()}= {avg_cost:.9f}, NDCG@{top_k}:{NDCG:.4f},HR@{top_k}:{HR:.4f}')




