import lightning
from torch.utils.data import DataLoader
from lhotse.utils import fix_random_seed
from lhotse.dataset import DynamicBucketingSampler

class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class DataModule(lightning.LightningDataModule):
    def __init__(
        self,
        train_ds,
        valid_ds,
        num_workers = 8,
        train_max_duration = 100,
        valid_max_duration = 100,
        quadratic_duration = 15,
        max_cuts = 32,
        shuffle = True,
        num_buckets = 20,
        bucket_buffer_size = 20000,
        shuffle_buffer_size = 10000,
        seed = 42,
    ):
        super().__init__()
        self.train_ds = train_ds
        self.valid_ds = valid_ds

        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_max_duration = train_max_duration
        self.valid_max_duration = valid_max_duration
        self.quadratic_duration= quadratic_duration
        self.max_cuts = max_cuts
        self.num_buckets = num_buckets
        self.bucket_buffer_size = bucket_buffer_size
        self.shuffle_buffer_size = shuffle_buffer_size
        self.seed = seed
        

    def train_dataloader(self):
        train_cuts_data = self.train_ds.cuts_data

        world_size = 1
        rank = 0

        # for lightning ddp training
        if self.trainer is not None:
            world_size = self.trainer.world_size
            rank = self.trainer.global_rank

        train_sampler = DynamicBucketingSampler(
            train_cuts_data,
            shuffle = self.shuffle,
            drop_last = False,
            max_duration = self.train_max_duration,
            quadratic_duration = self.quadratic_duration,
            max_cuts = self.max_cuts,
            num_buckets = self.num_buckets,
            buffer_size = self.bucket_buffer_size,
            shuffle_buffer_size = self.shuffle_buffer_size,
            seed = self.seed,
            world_size = world_size,
            rank = rank,
        )

        worker_init_fn = _SeedWorkers(self.seed)

        return DataLoader(
            self.train_ds,
            sampler = train_sampler,
            batch_size = None,
            num_workers = self.num_workers,
            persistent_workers = True,
            worker_init_fn = worker_init_fn,
        ) 

    def val_dataloader(self):
        valid_cuts_data = self.valid_ds.cuts_data
    
        world_size = 1
        rank = 0    

        valid_sampler = DynamicBucketingSampler(
            valid_cuts_data,
            shuffle = False,
            drop_last = False,
            max_duration = self.valid_max_duration,
            num_buckets = self.num_buckets,
            buffer_size = self.bucket_buffer_size,
            shuffle_buffer_size = self.shuffle_buffer_size,
            seed = self.seed,
            world_size=world_size,
            rank=rank,
        )

        return DataLoader(
            self.valid_ds,
            sampler = valid_sampler,
            batch_size = None,
            num_workers = 2
        )