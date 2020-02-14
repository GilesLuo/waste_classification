import argparse
import torch
# import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
from tqdm import tqdm
from datagenerator import ReadDataSource

from model import Model
import pdb

# setting gpu
# CUDA_VISIBLE_DEVICES = 1


# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_dir', type=str, default='../BA_dataset')
    args = parser.parse_args()

    train_dir = args.data_dir + '/joints_train_img'
    val_dir = args.data_dir + '/joints_val_img'
    train_csv = args.data_dir + '/train_set.csv'
    val_csv = args.data_dir + '/val_set.csv'

    print('loading training dataset ...')

    # train_dataset = ReadDataSource(x_dir=val_dir, y_file=val_csv)
    train_dataset = ReadDataSource(x_dir=train_dir, y_file=train_csv)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

    print('loading testing dataset ...')

    test_dataset = ReadDataSource(x_dir=val_dir, y_file=val_csv)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    print('loading model ...')

    model = Model()
    model.cuda()
    model.train()

    criterion = nn.CrossEntropyLoss()
    # criterion = nn.L1Loss()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    epoch_size = len(train_dataset) // args.batch_size

    print('------ start training ------')

    with open('log.txt', 'a') as f:
        f.write('Let\'s begin!')

    for epoch in range(args.num_epoch):

        model.train()
        batch_iterator = iter(train_loader)
        progress_bar = tqdm(range(epoch_size))
        losses = 0.
        count = 0

        for i in progress_bar:
            x, y = next(batch_iterator)
            x, y = x.cuda(), y.cuda()
            outs = model(x)
            # y = y.float()

            loss = criterion(outs, y)
            losses += loss.item() * x.size(0)
            count += x.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('epoch {:3d} avg cross entropy loss {:10.6f}'.format(epoch, losses / count))
        with open('log.txt', 'a') as f:
            f.write('\nepoch {:3d} avg loss {:10.6f}'.format(epoch, losses / count))

        with torch.no_grad():  # add

            model.eval()
            val_loss = 0.
            count = 0.
            correct = 0.
            error = 0.

            for idx, (x, y) in enumerate(tqdm(test_loader)):
                x, y = x.cuda(), y.cuda()  # x: [batch_size, 14, 48, 48]
                outs = model(x)

                y_f = y.float()
                loss = criterion(outs, y)
                val_loss += loss.item() * x.size(0)
                count += x.size(0)
                _, preds = outs.max(1)
                correct += preds.eq(y).sum()

                # MAE
                error += float(abs(preds - y).sum())

        mae = error / count

        acc = correct.float() / count

        print('Check! acc {:10.6f}  avg loss {:10.6f}'.format(acc, val_loss / count))
        print('MAE = {:10.6f}'.format(mae))
        with open('log.txt', 'a') as f:
            f.write('\nCheck! acc {:10.6f}  avg loss {:10.6f}'.format(acc, val_loss / count))
            f.write('\nMAE = {:10.6f}'.format(mae))

        savepoint = "./models/model_ep" + str(epoch) + ".pth.tar"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
            },
            savepoint
        )


if __name__ == '__main__':
    main()
