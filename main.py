from model import CPM2DPose
import torch
import numpy as np
import torchvision
import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import cv2

from glob import glob
import pandas

device = 'cuda:0'
num_joints = 21


class ObmanDataset(Dataset):
    def __init__(self, method=None, image_name=None):
        self.root = '/home/azatkariuly/CW4/obman_dataset/' #Change this path
        self.x_data = []
        self.y_data = []
        if method == 'train':
            self.root = self.root + 'train/'
            self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

        elif method == 'test':
            self.root = self.root + 'test/'

            if image_name:
              self.img_path = sorted(glob(self.root + 'rgb/' + image_name + '.jpg')) #we need it for 1.1
            else:
              self.img_path = sorted(glob(self.root + 'rgb/*.jpg'))

        for i in tqdm.tqdm(range(len(self.img_path))):
            img = cv2.imread(self.img_path[i], cv2.IMREAD_COLOR)
            print(self.img_path[i])
            b, g, r = cv2.split(img)
            img = cv2.merge([r, g, b])
            self.x_data.append(img)

            num = self.img_path[i].split('.')[0].split('/')[-1]
            img_pkl = self.root + 'meta/' + str(num) + '.pkl'
            pkl = pandas.read_pickle(img_pkl)
            coords_2d = pkl['coords_2d']
            self.y_data.append(coords_2d)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        transform1 = torchvision.transforms.ToTensor()
        new_x_data = transform1(self.x_data[idx])

        return new_x_data, self.y_data[idx]

class Trainer(object):
    def __init__(self, dataset, batchSize, epochs, loss='MSE'):
        self.batch_size = batchSize
        self.epochs = epochs

        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        if loss=='MSE':
            self.loss_fn = torch.nn.MSELoss()
        elif loss=='CrE':
            self.loss_fn = torch.nn.CrossEntropyLoss()
        elif loss=='MAE':
            self.loss_fn = torch.nn.L1Loss()
        elif loss=='Huba':
            self.loss_fn = torch.nn.SmoothL1Loss()

    def skeleton2heatmap(self, _heatmap, keypoint_targets):
        heatmap_gt = torch.zeros_like(_heatmap, device=_heatmap.device)

        keypoint_targets = (((keypoint_targets)) // 8)
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                x = int(keypoint_targets[i, j, 0])
                y = int(keypoint_targets[i, j, 1])
                heatmap_gt[i, j, x, y] = 1

        heatmap_gt = heatmap_gt.detach().cpu().numpy()
        for i in range(keypoint_targets.shape[0]):
            for j in range(21):
                heatmap_gt[i, j, :, :] = cv2.GaussianBlur(heatmap_gt[i, j, :, :], ksize=(3, 3), sigmaX=2, sigmaY=2) * 9 / 1.1772
        heatmap_gt = torch.FloatTensor(heatmap_gt).to(device)
        return heatmap_gt

    def train(self, poseNet, optimizer, epoch):

        #switch to train mode
        poseNet.train()

        losses = []

        for batch_idx, samples in enumerate(self.dataloader):
            x_train, y_train = samples
            heatmapsPoseNet = poseNet(x_train.cuda())
            gt_heatmap = self.skeleton2heatmap(heatmapsPoseNet, y_train)

            loss = self.loss_fn(heatmapsPoseNet, gt_heatmap)

            #optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            losses.append(loss.item())

            ## Print train result
            if batch_idx % 20 == 0:
                print('Epoch {:4d}/{} Batch {}/{} Loss {}'.format(
                    epoch, self.epochs, batch_idx, len(self.dataloader), np.array(losses).mean()
                ))

        return poseNet

class Tester(object):
    def __init__(self, dataset, batch_size):
        self.batch_size = batch_size

        #dataset = ObmanDataset(method='test')
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        self.datalen = dataset.__len__()
        self.mse_all_img = []

    def heatmap2skeleton(self, heatmapsPoseNet):
        skeletons = np.zeros((heatmapsPoseNet.shape[0], heatmapsPoseNet.shape[1], 2))
        for m in range(heatmapsPoseNet.shape[0]):
            for i in range(heatmapsPoseNet.shape[1]):
                u, v = np.unravel_index(np.argmax(heatmapsPoseNet[m][i]), (32, 32))
                skeletons[m, i, 0] = u * 8
                skeletons[m, i, 1] = v * 8
        return skeletons

    def test(self, poseNet):
        error = []

        #switch to evaluate mode
        poseNet.eval()

        for batch_idx, samples in enumerate(self.dataloader):
            x_test, y_test = samples
            heatmapsPoseNet = poseNet(x_test.cuda()).cpu().detach().numpy()
            skeletons_in = self.heatmap2skeleton(heatmapsPoseNet)

            for i in range(skeletons_in.shape[0]):
              err = self.calc_error(skeletons_in[i], y_test[i])

              error.append(err)

        total_error = np.array(error).mean()

        #print error
        print('Total Error = ', total_error)

        return total_error

    def calc_error(self, preds, gts):
      error = 0

      K = preds.shape[0] #K=21

      for i in range(K):
        pred = preds[i]
        gt = gts[i].tolist()

        res = ((pred[0]-gt[0])**2 + (pred[1]-gt[1])**2)**0.5
        error += res

      error = error/K

      return error

def main():

    epochs = 100
    batchSize = 16
    learningRate = 1e-2

    #don't touch
    best_error = -1

    #initialize the network
    poseNet = CPM2DPose()
    poseNet = poseNet.to(device)

    #initialize the Dataset
    train_dataset = ObmanDataset('train')
    test_dataset = ObmanDataset('test')

    # Load of pretrained_weight file
    weight_root = train_dataset.root.split('/')
    del weight_root[-2]
    weight_root = "/".join(weight_root)
    weight_PATH = weight_root + 'pretrained_model/pretrained_weight.pth'
    poseNet.load_state_dict(torch.load(weight_PATH))

    #start
    optimizer = torch.optim.SGD(poseNet.parameters(), lr=learningRate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs,  eta_min=0, last_epoch=-1)

    #start Training
    trainer = Trainer(train_dataset, batchSize, epochs, loss='Huba')
    tester = Tester(test_dataset, batchSize)

    date = '201105'

    for epoch in tqdm.tqdm(range(epochs + 1)):

        #if epoch%30 == 0:
        #    learningRate = learningRate / 10
        #    optimizer = torch.optim.SGD(poseNet.parameters(), lr=learningRate)

        print('Training...')
        print('learningRate =', scheduler.get_lr())

        poseNet = trainer.train(poseNet, optimizer, epoch)

        print('Testing...')

        total_error = tester.test(poseNet)

        if best_error == -1:
            best_error = total_error
            torch.save(poseNet.state_dict(), "_".join(['/home/azatkariuly/CW4/', date, 'best_model_2_not.pth']))
        else:
            if total_error < best_error:
                best_error = total_error
                torch.save(poseNet.state_dict(), "_".join(['/home/azatkariuly/CW4/', date, 'best_model_2_not.pth']))

        print('THE BEST:', best_error)
        scheduler.step()


if __name__ == '__main__':
    main()

