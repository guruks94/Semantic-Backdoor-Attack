import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

import torchvision.transforms as transforms
import torchvision.models as models

import os
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_auc_score


class BackdoorDataset(Dataset):

    def __init__(self, images_path, file_path, augment=None):
        self.img_list = []
        self.img_label = []
        self.augment = augment

        with open(file_path, "r") as fileDescriptor:
            line = fileDescriptor.readline()

            while line:
                line = fileDescriptor.readline()

                if line:
                    # lineItems = line.split()
                    lineItems = line.split(',')
                    imagePath = os.path.join(images_path, lineItems[0])
                    imageLabel = [int(lineItems[1])]
                    # imageLabel = lineItems[1:num_class + 1]
                    # imageLabel = [int(i) for i in imageLabel]

                    self.img_list.append(imagePath)
                    self.img_label.append(imageLabel)

    def __getitem__(self, index):
        imagePath = self.img_list[index]
        imageData = Image.open(imagePath).convert('RGB')
        imageLabel = torch.FloatTensor(self.img_label[index])

        if self.augment != None:
            imageData = self.augment(imageData)

        return imageData, imageLabel

    def __len__(self):

        return len(self.img_list)


def build_transform_classification(normalize, crop_size=224, resize=256, mode="train"):
    transformations_list = []

    if normalize.lower() == "imagenet":
        normalize = transforms.Normalize(
            [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    else:
        normalize = None

    if mode == "train":
        transformations_list.append(transforms.RandomResizedCrop(crop_size))
        transformations_list.append(transforms.RandomHorizontalFlip())
        transformations_list.append(transforms.RandomRotation(7))
        transformations_list.append(transforms.ToTensor())
        if normalize is not None:
            transformations_list.append(normalize)
    elif mode == "test":
        transformations_list.append(transforms.Resize((resize, resize)))
        transformations_list.append(transforms.TenCrop(crop_size))
        transformations_list.append(
            transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        if normalize is not None:
            transformations_list.append(transforms.Lambda(
                lambda crops: torch.stack([normalize(crop) for crop in crops])))
    transformSequence = transforms.Compose(transformations_list)

    return transformSequence


def classification_engine(train_set, num_class, learning_rate, is_pretrained, batch_size, path_to_save_model, model_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    cudnn.benchmark = True

    save_model_path = os.path.join(path_to_save_model, model_name + ".pth")

    data_loader_train = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    #model = models.resnet50(pretrained=True)
    model = models.__dict__[model_name](pretrained=is_pretrained)
    #print(model)

    for param in model.parameters():
        param.requires_grad = False

    if model_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif model_name.lower().startswith("efficientnet"):
        kernelCount = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print("Enter correct model name")
        return

    criterion = nn.BCELoss()
    model.to(device)

    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

    # for param in model.parameters():
    #   print(param)

    for epoch in range(start_epoch, end_epoch):
        print(f"epoch: {epoch}")
        model.train()
        running_loss = 0
        for i, (samples, targets) in enumerate(data_loader_train):
            samples, targets = samples.float().to(device), targets.float().to(device)
            optimizer.zero_grad()
            outputs = model.forward(samples)
            # print(f"outputs: {outputs}")
            # print(f"targets: {targets}")
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 5 == 0:
                print(f"batch[{i}]: loss={loss.item()}")

        print(f"epoch[{epoch}]: Total loss={running_loss}")
    torch.save(model.state_dict(), save_model_path)
    
    # To save the weights to use on single gpu if trained on multiple gpu's
    #torch.save(model.module.state_dict(), save_model_path) 
    # for param in model.parameters():
    #   print(param)


def test_classification(test_set, num_class, learning_rate, is_pretrained, batch_size, saved_model, model_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    
    data_loader_test = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    model = models.__dict__[model_name](pretrained=is_pretrained)
    #model = models.resnet50(pretrained=False)

    for param in model.parameters():
        param.requires_grad = False
        
    if model_name.lower().startswith("resnet"):
        kernelCount = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=learning_rate)
    elif model_name.lower().startswith("efficientnet"):
        kernelCount = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Linear(kernelCount, num_class), nn.Sigmoid())
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        print("Enter correct model name")
        return

    criterion = nn.BCELoss()

    if torch.cuda.device_count() > 1:
      model = torch.nn.DataParallel(model)

    model.load_state_dict(torch.load(saved_model))

    model.to(device)
    model.eval()

    y_test = torch.FloatTensor().to(device)
    p_test = torch.FloatTensor().to(device)

    with torch.no_grad():
        for i, (samples, targets) in enumerate(tqdm(data_loader_test)):
            print(f"{i}th batch")
            print(f"samples: {samples.shape}")

            samples, targets = samples.float().to(device), targets.float().to(device)
            
            y_test = torch.cat((y_test, targets), 0)

            if len(samples.size()) == 4:
                bs, c, h, w = samples.size()
                n_crops = 1
            elif len(samples.size()) == 5:
                bs, n_crops, c, h, w = samples.size()

            varInput = torch.autograd.Variable(samples.view(-1, c, h, w))
            # print(f"varInput: {varInput.shape}")

            outputs = model.forward(varInput.to(device))
            # print(f"outputs: {outputs.shape} {outputs}")

            outMean = outputs.view(bs, n_crops, -1).mean(1)
            # print(f"outMean: {outMean.shape} {outMean}")

            p_test = torch.cat((p_test, outMean.data), 0)
            # print(f"p_test: {p_test.shape} {p_test}")

            # mask = p_test > 0.1
            # print(f"mask: {mask}")

            # new_samples = torch.FloatTensor().cuda()

            # for i, val in enumerate(mask):
            #   if val.item():
            #     new_samples = torch.cat((new_samples, samples[i]), 0)

            # new_samples = new_samples.view(-1, n_crops, c, h, w)

            # print(f"new_samples: {new_samples.shape}")

            # break
    return y_test, p_test


def metric_AUROC(target, output, num_class):
    outAUROC = []

    target = target.cpu().numpy()
    output = output.cpu().numpy()

    for i in range(num_class):
        outAUROC.append(roc_auc_score(target[:, i], output[:, i]))

    return outAUROC


if __name__ == "__main__":

    # print(torch.cuda.is_available())
    # print(torch.cuda.device_count())
    # print(torch.cuda.current_device())
    # print(torch.cuda.device(0))
    # print(torch.cuda.get_device_name(0))

    # Define the path here
    # Local Mac path
    # image_path = "/Users/gpks/Documents/ASU/06Fall2022/CSE598/Project/Semantic_Backdoor_Attack/data"
    # train_list = "/Users/gpks/Documents/ASU/06Fall2022/CSE598/Project/Semantic_Backdoor_Attack/dataset/train_less.csv"
    # test_list = "/Users/gpks/Documents/ASU/06Fall2022/CSE598/Project/Semantic_Backdoor_Attack/dataset/test_less.csv"
    # path_to_save_model = "/Users/gpks/Documents/ASU/06Fall2022/CSE598/Project/Semantic_Backdoor_Attack/models/less"

    # Agave path
    image_path = "/home/gkempego/CSE598/data"
    train_list = "/home/gkempego/CSE598/dataset/train_clean.csv"
    test_list = "/home/gkempego/CSE598/dataset/test_clean.csv"
    # To test poisoned test images on poisoned model
    test_list_pd = "/home/gkempego/CSE598/dataset/test_pd.csv"
    path_to_save_model = "/home/gkempego/CSE598/models/clean/run1"

    # Set the hyper parameters for the model
    num_class = 1
    learning_rate = 0.001
    batch_size = 128
    start_epoch = 0
    end_epoch = 10

    is_pretrained = True
    model = "resnet50"

    train_set = BackdoorDataset(images_path=image_path, file_path=train_list, augment=build_transform_classification(normalize="imagenet", mode="train"))
    test_set = BackdoorDataset(images_path=image_path, file_path=test_list, augment=build_transform_classification(normalize="imagenet", mode="test"))
    test_set_pd = BackdoorDataset(images_path=image_path, file_path=test_list_pd, augment=build_transform_classification(normalize="imagenet", mode="test"))

    print ("start training.....")
    classification_engine(train_set, num_class, learning_rate, is_pretrained, batch_size, path_to_save_model, model_name=model)

    saved_model = os.path.join(path_to_save_model, model + ".pth")
    # saved_model = "/home/gkempego/CSE598/models/clean/resnet50.pth"
    # saved_model = "/Users/gpks/Documents/ASU/06Fall2022/CSE598/Project/Semantic_Backdoor_Attack/models/clean/resnet50.pth"

    print ("start testing on clean data.....")
    y50_test, p50_test = test_classification(test_set, num_class, learning_rate, is_pretrained, batch_size, saved_model, model_name=model)
    AUC_value = metric_AUROC(y50_test, p50_test, num_class)
    print(f"Clean model on clean test data - AUC: {AUC_value}")

    print ("start testing on poisoned data.....")
    y50_test_pd, p50_test_pd = test_classification(test_set_pd, num_class, learning_rate, is_pretrained, batch_size, saved_model, model_name=model)
    AUC_value_pd = metric_AUROC(y50_test_pd, p50_test_pd, num_class)
    print(f"Clean model on posioned test data - AUC: {AUC_value_pd}")
