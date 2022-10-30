import numpy as np
from numpy.random import permutation



def train_test_split(x, y, test_size):
    permut_index = np.random.permutation(x.shape[0])
    test_index = permut_index[:test_size]
    train_index = permut_index[test_size:]
    test_x = x[test_index]
    test_y = y[test_index]
    train_x = x[train_index]
    train_y = y[train_index]
    return train_x, test_x, train_y, test_y


def onehot(x: np.ndarray, num_classes):
    return np.eye(num_classes)[x]


def classify_accuracy(y_pred, y_true):
    assert y_pred.any() != np.nan
    N = float(y_true.shape[0])
    y_true = np.argmax(y_true, axis=1)
    return np.sum(y_true == np.argmax(y_pred, axis=1)) / N


class Dataset:
    def __init__(self) -> None:
        pass

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class Sampler:
    def __init__(self, dataset: Dataset) -> None:
        pass

    def __iter__(self):
        raise NotImplementedError


class SequentialSampler(Sampler):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        return iter(range(len(self.dataset)))

    def __len__(self):
        return len(self.dataset)


class RandomSampler(Sampler):
    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __iter__(self):
        yield from permutation(len(self.dataset)).tolist()

    def __len__(self):
        return len(self.dataset)


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int,
                 drop_last: bool) -> None:
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class _DataLoaderIter:
    def __init__(self, loader) -> None:
        self.loader = loader
        self.sample_iter = iter(self.loader.batch_sampler)

    def __next__(self):
        index = next(self.sample_iter)
        return self.loader.dataset[index]


class DataLoader:
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = False,
                 drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        if shuffle:
            self.sampler = RandomSampler(dataset)
        else:
            self.sampler = SequentialSampler(dataset)

        self.batch_sampler = BatchSampler(self.sampler, batch_size, drop_last)
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        return _DataLoaderIter(self)

#dataloader
def data_loader(X, y, batch_size: int, shuffle: bool = False) -> list:
    class TrainSet(Dataset):
        def __init__(self, X, y) -> None:
            self.data = X
            self.target = y

        def __getitem__(self, index):
            return self.data[index], self.target[index]

        def __len__(self):
            return len(self.data)

    return DataLoader(TrainSet(X, y), batch_size, shuffle)

# 处理反向传播时ndim不匹配的函数
def process_grad(add_grad: np.ndarray, node: np.ndarray):
    if add_grad.shape != node.shape:
        add_grad = np.sum(
            add_grad,
            axis=tuple(-i for i in range(1, node.ndim + 1)
                       if node.shape[-i] == 1),
            keepdims=True,
        )
        add_grad = np.sum(
            add_grad,
            axis=tuple(range(add_grad.ndim - node.ndim)),
        )
    return add_grad
