import torch.nn as nn

from torchvision import datasets, transforms, models
from torchvision.transforms.functional import InterpolationMode
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F



imagenet_origin_transform = transforms.Compose([
                                    transforms.Resize(
                                        size=256,
                                        interpolation=InterpolationMode.BILINEAR,
                                    ),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                ])

CUB_dataset = datasets.ImageFolder('CUB_200_2011/CUB_200_2011/images', 
                                    imagenet_origin_transform)

def train(device, train_loader, model, optimizer, scheduler=None):
    correct = 0
    total = 0

    for batch_idx, batch in enumerate(train_loader):
        im, target, _ = batch
        total+= im.shape[0]

        im, target = im.to(device), target.to(device)
        optimizer.zero_grad()
        
        output = model(im)
        
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
        ce_loss = F.cross_entropy(output, target)
        loss = ce_loss

        total_loss += loss
        loss.backward()
        optimizer.step()
    if scheduler != None:
        scheduler.step()

    total_loss = float(total_loss)
    accuracy = float(correct.item()/total)
    return total_loss, accuracy

def eval(model, val_loaders, device):
    class_correct = 0
    for batch_idx, batch in enumerate(val_loaders):
        im, target, _ = batch
        im, target = im.to(device), target.to(device)
        output = model(im)
        pred = output.data.max(1, keepdim=True)[1]
        class_correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    return class_correct/len(val_loaders.dataset)

def transferbility(model, dataset, device):
    with open("result.txt", "w") as f:
        f.write("caution, when ever you run the file, it clears the content at first")
    CUB_train, CUB_test = torch.utils.data.random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(CUB_train, batch_size=32, shuffle=True)
    test_loader = DataLoader(CUB_test, batch_size=32, shuffle=False)


    for param in model.parameters():
        param.requires_grad = False
    model.fc = torch.nn.Linear(2048, 200)
    model.eval()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    for epoch in range(60):
        _, acc = train(device, train_loader, model, scheduler)
        eval_acc = eval(model, test_loader, device)
        print(f"train_acc: {acc}, val_acc: {eval_acc}")
        with open("result.txt", "a") as f:
            f.write(f"epoch{epoch}: train-{acc}, val-{eval_acc}")

if __name__ == "__main__":
    transferbility(models.resnet50(pretrained=True), CUB_dataset, torch.device("cuda:0"))