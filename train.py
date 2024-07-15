#!~/anaconda3/envs/yoga/bin/python
#! export TORCH_DISTRUBITED_DEBUG=INFO
import torch
import torch.distributed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
import os
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import random


from dataset import VideoDataset
from models import CNN_LSTM_clip, CNN_LSTM_frame
from tqdm import tqdm
from utils._tool import  split_dataset, visualize_test_results
import time

def cleanup():
    torch.distributed.destroy_process_group()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='dataset/OD_dataset', type=str, help='frames folder path')
    parser.add_argument('--save_path', default='output/test_results/OD/test', type=str, help='save folder path')
    parser.add_argument('--height', default=112, type=int, help='height of the input image')
    parser.add_argument('--width', default=112, type=int, help='width of the input image')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
    parser.add_argument('--cnn_net', default='resnet18', type=str, help='resnet18, resnet34, resnet50')
    parser.add_argument('--num_classes', default=8, type=int, help='number of classes')
    parser.add_argument('--cnn_hidden_size', default=2048, type=int, help='hidden size of the CNN')
    parser.add_argument('--lstm_hidden_size', default=256, type=int, help='hidden size of the LSTM')
    # parser.add_argument('--gpu', default='cuda:1', type=str, help='gpu id')
    parser.add_argument('--local-rank', default=-1, type=int, help='local rank for distributed training')

    return parser.parse_args()


def preprocess(height, width, data_path):
    
    data_transform = transforms.Compose([
        transforms.Resize((height, width)),     #Resize
        # transforms.ToTensor()              #Convert to Pytorch tensors;  only frame model need this
    ])

    dataset = VideoDataset(root_dir=data_path, transform=data_transform)
    
    return dataset


def pad_collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    return padded_sequences, torch.tensor(labels), torch.tensor(lengths, dtype=torch.long)

def training(model, train_dataloader, valid_dataloader, batch_size, cnn_net, rank):

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()

    # Training loop
    num_epochs = 100
    if rank == 0:
        writer = SummaryWriter(f'./run_clips/{cnn_net}_LSTM_batch{batch_size}', comment='ddp-training')

    
    with tqdm(total=num_epochs, desc='Training', ncols=10,  leave=True) as pbar:
        for epoch in range(num_epochs):
            # print(f'Epoch {epoch+1}/{num_epochs}')
            # training

            model.train()
            running_loss = 0.0
            total_train = 0
            correct_train = 0
            
            start_time = time.time()
        
            for batch_idx, (inputs, labels, lenghts) in enumerate(train_dataloader):
                optimizer.zero_grad()
                inputs, labels = inputs.to(device), labels.to(device)
                lenghts = lenghts.cpu()
                
                with autocast():
                    outputs = model(inputs, lenghts)
                    loss = criterion(outputs, labels)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()


                _, predicted = outputs.max(1)
                total_train += labels.size(0)
                correct_train += predicted.eq(labels).sum().item()

                # if batch_idx % 10 == 9 and rank==0:  # Log every 10 batches
                if rank==0:
                    writer.add_scalar('Training Loss', running_loss/(batch_idx+1) , epoch * len(train_dataloader) + batch_idx)
                    writer.add_scalar('Training Accuracy', correct_train / total_train, epoch * len(train_dataloader) + batch_idx)
                    # running_loss = 0.0
                
                # pbar.set_postfix(loss= running_loss / (batch_idx+1), acc= correct_train / total_train)  # Assume you calculate train_loss per batch
                # pbar.update(1)

        
            torch.cuda.empty_cache()
            model.eval()
            validation_loss = 0.0
            correct_val = 0
            total_val = 0


            for batch_idx, (inputs_val, labels_val, lengths_val) in enumerate(valid_dataloader):
                inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
                lengths_val = lengths_val.cpu()#.to(device)
                
                with autocast():
                    outputs_val = model(inputs_val, lengths_val)
                    loss_val = criterion(outputs_val, labels_val)
                validation_loss += loss_val.item()

                _, predicted_val = outputs_val.max(1)
                total_val += labels_val.size(0)
                correct_val += predicted_val.eq(labels_val).sum().item()

                if rank==0:  # Log every 10 batches
                    writer.add_scalar('Validation Loss', validation_loss/(batch_idx+1), epoch * len(valid_dataloader) + batch_idx)
                    writer.add_scalar('Validation Accuracy', correct_val / total_val, epoch * len(valid_dataloader) + batch_idx)
                    running_loss = 0.0
                    # pbar.update(1)
                    # pbar.set_postfix(loss= running_loss / 10, acc= correct_train / total_train)  # Assume you calculate train_loss per batch

            avg_loss_val = validation_loss / len(valid_dataloader)
            accuracy_val = correct_val / total_val
            epoch_time = time.time() - start_time
            if rank==0:
                tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Val Loss: {avg_loss_val}, Val Accuracy: {accuracy_val}, Time: {epoch_time:.2f}s')
                pbar.update(1)

            if epoch % 10 == 9 and rank==0:
                model_path = f'./model/{cnn_net}_LSTM_batch{batch_size}_clip.pt'
                torch.save(model.state_dict(), model_path)
            
            # pbar.update(1)
        

        torch.cuda.empty_cache()

    if rank==0:
        writer.close()

    print("Training finished ", cnn_net)


def testing(model, test_dataloader, batch_size, cnn_net, save_dir):
    print(f'load ./model/{cnn_net}_LSTM_batch{batch_size}_clip.pt')
    model.load_state_dict(torch.load(f'./model/{cnn_net}_LSTM_batch{batch_size}_clip.pt'))

    visualize_test_results(model, test_dataloader, save_dir, device)

def main(rank):
    args = parse_args()
    data_path = args.data_path#os.path.join('home/viplab/yuru/yoga/', args.data_path)
    save_path =  args.save_path#os.path.join('home/viplab/yuru/yoga/', args.save_path)
    os.makedirs(save_path, exist_ok=True)
    # global device
    # device = torch.device(args.gpu if torch.cuda.is_available() else "cpu")

    dataset = preprocess(args.height, args.width, data_path)
    print(dataset.labels_to_index)
    # split_ratio = 0.8  #80% for training, 20% for valid
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, test_size=0.2, val_size=0.1)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=pad_collate_fn, num_workers=16)
    valid_sampler = DistributedSampler(valid_dataset)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, sampler=valid_sampler, collate_fn=pad_collate_fn, num_workers=16)
    test_sampler = DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, collate_fn=pad_collate_fn, num_workers=16)

    model = CNN_LSTM_clip(len(dataset.labels), args.cnn_hidden_size, args.lstm_hidden_size, args.cnn_net)
    # model = CNN_LSTM_frame(len(dataset.labels), args.cnn_hidden_size, args.lstm_hidden_size, args.cnn_net)
    # model = nn.DataParallel(model)
    model.to(rank)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    training(model, train_loader, valid_loader, args.batch_size, args.cnn_net, rank)
    testing(model, test_loader, args.batch_size, args.cnn_net, save_path)

    cleanup()

if __name__ =='__main__':
    world_size = torch.cuda.device_count
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.distributed.init_process_group(backend='nccl')
    
    set_seed(42)
    print(local_rank)
    main(local_rank)
    # torch.multiprocessing.spawn(main, args=(world_size), nprocs=world_size, join=True)
    # torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)

# model_path = './model/CNN_LSTM_batch32_crop.pt'

 # transforms.RandomHorizontalFlip(),
    # transforms.RandomRotation(degrees=30),
    # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),


    # test_dataloader = DataLoader(test_dataset, batch_size=batch, shuffle=False)
    # save_dir = './test_results'

    # # Test the model and save results
    # visualize_test_results(model, test_dataloader, save_dir)
    # criterion = nn.CrossEntropyLoss()
    # test_model(model, test_dataloader, criterion)
