from __future__ import print_function

import torch
import torch.nn as nn

import pandas as pd
import torch
import torch.utils.data
import torchvision
import pdb
from tqdm import tqdm


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    source: https://github.com/ufoym/imbalanced-dataset-sampler

    Arguments:
        indices: a list of indices
        num_samples: number of samples to draw
        callback_get_label: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(
        self,
        dataset,
        labels: list = None,
        indices: list = None,
        num_samples: int = None,
        callback_get_label = None,
    ):
        # if indices is not provided, all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) if indices is None else indices
        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided, draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        

        # distribution of classes in the dataset
        df = pd.DataFrame()
        df["label"] = self._get_labels(dataset) if labels is None else labels
        df.index = self.indices
        df = df.sort_index()

        label_to_count = df["label"].value_counts()

        weights = 1.0 / label_to_count[df["label"]]

        self.weights = torch.DoubleTensor(weights.to_list())

    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.dataset.imgs[:][1]
        elif isinstance(dataset, torch.utils.data.ConcatDataset): # added. add before next `elif` because ConcatDataset belong to torch.utils.data.Dataset

            return self._get_all_labels_by_min(dataset) # added
        # elif isinstance(dataset, torch.utils.data.Dataset):
        #     return [ self._roomreader_quantize_label_4class(torch.from_numpy(batch[0]['eng'][:,-1])).min().long().item()  for batch in dataset] # added
        else:
            return self._get_all_labels_by_min(dataset)
            print('here')


    def _get_all_labels_by_min(self, dataset):
        loader = torch.utils.data.DataLoader(dataset, batch_size = len(dataset), shuffle = False, num_workers = 0)
        for idx, batch in enumerate(tqdm(loader)): pass
        return batch['reaction_visual'].long().squeeze().tolist()
            
    def _roomreader_quantize_label_4class(self,target):
        target = (target + 2)
        target = torch.clip(target, min = 0, max = 3)
        target = torch.floor(target)

        return target

    def _roomreader_quantize_vel_label_4class(self,target):
        target = torch.bucketize(target, torch.tensor([-4 , -1, 0, 1, 4]))

        return target 


    def _get_concat_labels(self,concatdataset): # added
        dataset_list = concatdataset.datasets
        concat_labels = []
        for ds in dataset_list:
            concat_labels.extend(ds.get_labels())
        return concat_labels

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

nan_list = ['c0c54a77-1d33-41a4-8e13-92a4840e82b8',
 '60454fcf-eceb-4faf-9347-8d796e1b5be8',
 '8d0c52af-1e6d-46ca-a709-28c7ba9734ce',
 '588c5b4b-5e92-426c-8acc-686628a7342f',
 '6ce9f678-15c4-424c-bf3f-b4e2f47a8818',
 '29f8f496-079b-4a71-84ff-7100bbc28824',
 'fe4a5de5-3b9c-4b6f-8e70-403db8a1caec',
 '115ba192-7e26-497b-85b2-adf378b33387',
 '65b9bc57-6206-475d-91ec-c5696259c0d4',
 '446fe8dc-1619-4c44-a4bd-8cbb48a2bca1',
 '0278950b-a7e0-4e15-8a2b-1629ff1b17ba',
 '542a6af4-84f3-4681-80ce-fce29162efc1',
 '3049cb21-cec9-46b6-ace5-8b97f4fd6165',
 '82af78ef-4d4f-4bd3-8d20-7daa70adff61',
 '65ec7b23-af77-449e-9b11-e431f8a3b874',
 '32adb5d5-910d-4547-972a-f5d0b795c689',
 '5f3c5c14-1280-48c4-b6a5-97437ea68c94',
 'bf57c9e7-7be9-4961-a7fb-59777c0dc751',
 'debcfb81-d883-4fb5-8c4c-cc98468c96db',
 'b91c719a-4d41-4e24-837e-5c6abfacc77a',
 'c4caca36-1ace-44de-844a-0933eface36a',
 '6fdd1fc5-e185-45e4-810d-bd8fb8b82490',
 'ceaafa07-24d8-4398-9b1a-b825938f23e0',
 'a8855c03-359f-42d9-af04-e459f9547107',
 '98666aa9-2a23-4a64-9379-48541d73d901',
 '030c76ab-9e19-4b78-9a7a-86cd7ba8472a',
 '601dd44f-db11-48d6-b150-d8959e05c97f',
 '17dbcae3-0087-49c6-af7c-c92099e3377a',
 'fe2ec1ed-027e-404d-94c7-8fc1587aa3bc',
 '49694675-cacf-452d-a940-3c93987126ef',
 '22579311-0848-472a-a1b7-9b663fbb4aab',
 '1e3c22d6-422c-4921-8892-e31e09f9a2f6',
 '87e8b3ce-14da-43e6-bd8b-7c3c67030559',
 'eb326986-325e-4f6b-b895-82a1f577c797',
 '92c66875-8a3c-44f6-9356-8537655417db',
 'a806bdc5-250b-41e7-a8c7-9440c270f3fe',
 'b877a5bf-3384-4ea0-bf82-cef399a1b00d',
 '983a1ff0-e14f-408a-a807-2e1b1cbb2a00',
 '80d9e496-db7f-49e3-8789-06850e62ffa1',
 '68a27e9e-2c9d-49ac-ba58-9751d402a84b',
 '0a84a137-b947-441c-b94c-a03f4a5851ea',
 'd9266679-7d71-43d0-ab1a-3293b589569e',
 '141ea746-d1f1-402d-9b0a-a4cdb0b1c4f0',
 '20bc98a3-efaa-4657-8448-f731bdec47cb',
 '2a4d7a05-b514-4927-a797-3644b4046f43',
 'd713d070-fe2c-4327-8952-e78f45d251a8',
 'da272ccd-9b89-4ae4-81d4-38ed452f36d1',
 '3045ec04-252c-420e-8646-c6b0e150ca74',
 'ee19d0ea-462c-47ef-888d-ac3254113e37',
 'f5de68ee-6513-406a-b0fc-49f20873faef',
 '53459a58-b890-4cad-83dc-1fb55dbd880e',
 'ea29afa5-e18a-47a9-93bf-87bfbb94c1d5',
 '13f6956b-ff2a-4ad3-aed6-8fd3cfdb2cd4',
 '765f6cde-5291-4047-89c1-d71b1e3a413d',
 'e7937025-762a-4415-b401-a6299e917420', #start of long list all the way down
 '777e03b5-425e-43eb-af4f-d8dbb313c319',
 'd9f18045-50ab-4584-9094-49bbedff4b72',
 '03d82c5a-c923-47de-90f8-621776ff6cbc',
 'f77a4b03-ad8f-438a-a85b-20c055aeed48',
 'f7d7f6ce-ff6e-4c5e-87dc-949399e0573b',
 '40d9ff29-4edc-4570-b980-fcdf70bb1629',
 '33d4bd03-1a0b-4252-9ebb-9ab864a57a7e',
 'adba8838-0c7e-4013-9734-46dfd60dd668',
 'ab642b9e-2060-4c49-86c3-92cfc0c865f0',
 'd777011c-80d2-4345-8977-e472d8ef479a',
 '07b732f9-a9e0-43e9-85a4-263a6d78af0d',
 '765f6cde-5291-4047-89c1-d71b1e3a413d',
 'c2bbe917-15c3-4d3b-98a4-866fcf48c303']


reward_dict = {'mean': 73.78779840848806,
               'std': 20.11304417738341}
