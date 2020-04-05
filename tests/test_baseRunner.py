import covid_county_prediction.BaseRunner as BaseRunner
import torchvision.models as models
import torch.nn as nn
import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

#inspired from https://github.com/pytorch/examples/blob/master/imagenet/main.py
class TestBaseRunner(BaseRunner.BaseRunner):
    def __init__(self, models, loss_fn, optimizers, best_metric_name = 'acc1',
        should_minimize_best_metric = False, debug = True):
        super(TestBaseRunner, self).__init__(models, loss_fn, optimizers, 
            best_metric_name, should_minimize_best_metric, debug)

    def train_batch_and_get_metrics(self, batch):
       out = self.nets[0](batch[0])
       loss = self.loss_fn(out, batch[1])       
       acc1, acc5 = self.accuracy(out, batch[1], topk=(1, 5))

       self.optimizers[0].zero_grad()
       loss.backward()
       self.optimizers[0].step()

       return [('loss', loss.mean().item()), ('acc1', acc1), ('acc5', acc5)]

    def test_batch_and_get_metrics(self, batch):
        out = self.nets[0](batch[0])
        loss = self.loss_fn(out, batch[1])
        acc1, acc5 = self.accuracy(out, batch[1], topk=(1, 5))

        return [('loss', loss.mean().item()), ('acc1', acc1), ('acc5', acc5)]

    def accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        #taken from https://github.com/pytorch/examples/blob/master/imagenet/main.py
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
            return res

if(__name__ == '__main__'):
    torch.set_default_tensor_type(torch.cuda.FloatTensor 
        if torch.cuda.is_available() else torch.FloatTensor)

    net = models.vgg11()
    net.classifier[-1] = nn.Linear(net.classifier[-1].in_features, 10)

    loss_fn = nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        torch.cuda.set_device('cuda:0')
        net = net.cuda()
        loss_fn = loss_fn.cuda()

    optimizer = torch.optim.SGD(net.parameters(), 0.01,
                                momentum=0.9,
                                weight_decay=5e-4)

    runner = TestBaseRunner([net], loss_fn, [optimizer])

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.4914, 0.4822, 0.4465], 
            [0.2023, 0.1994, 0.2010])])

    train_data   = datasets.CIFAR10('./', download=True, train=True, transform=transform)
    test_data   = datasets.CIFAR10('./', train=False, transform=transform)

    train_loader = data.DataLoader(train_data, shuffle=True, batch_size=128)
    test_loader = data.DataLoader(test_data, shuffle=True, batch_size=512)

    runner.train(train_loader, 5, test_loader)
    runner.test(test_loader, validate=True)
