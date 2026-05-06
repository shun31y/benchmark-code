from torch.utils.data import DataLoader
from data_loaders.datasets_composition import CompMDMUnfoldingGeneratedDataset
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


# our loader
def get_mdm_loader(args, model, diffusion, batch_size, eval_file, dataset, max_motion_length=200,
                   precomputed_folder=None, scenario=""):
    opt = {
        'name': 'test',  # FIXME
    }
    print('Generating %s ...' % opt['name'])
    generation_batch_size = getattr(args, "generation_batch_size", getattr(args, "eval_batch_size", batch_size))
    dataset = CompMDMUnfoldingGeneratedDataset(args, model, diffusion, max_motion_length, eval_file, w_vectorizer=dataset.w_vectorizer, opt=dataset.opt,
                                            precomputed_folder=precomputed_folder, scenario=scenario,
                                            generation_batch_size=generation_batch_size)
    mm_motion_loader = None

    # NOTE: bs must not be changed! this will cause a bug in R precision calc!
    num_workers = 0
    motion_loaders = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=num_workers > 0,
    )

    print('Generated Dataset Loading Completed!!!')

    return motion_loaders, mm_motion_loader
