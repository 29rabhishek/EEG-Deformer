from models.model_handler import DLModel
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from utils import *
from utils_clf import create_dataset
from torchvision import transforms

def LOSO(experiment_ID, logs_name, args):
    pl.seed_everything(seed=args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.backends.cudnn.deterministic = True

    image_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
 
    train_dataloader, val_dataloader = create_dataset(train_batch_size = args.batch_size,transform=image_transform)

    
    model = DLModel(config=args)
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode='max')
    ensure_path(args.save_path)
    logger = TensorBoardLogger(save_dir=args.save_path, version=experiment_ID, name=logs_name)
    # most basic trainer, uses good defaults (1 gpu)
    if args.mixed_precision:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback], precision='16-mixed',
            limit_val_batches = 0.3,
            limit_test_batches = 0.1
        )
    else:
        trainer = pl.Trainer(
            accelerator="gpu", devices=[args.gpu], max_epochs=args.max_epoch, logger=logger,
            callbacks=[checkpoint_callback],
            limit_val_batches = 0.3,
            limit_test_batches = 0.1
        )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    best_val_metrics = trainer.checkpoint_callback.best_model_score.item()
    results = trainer.test(ckpt_path="best", dataloaders=val_dataloader)
    results[0]['best_val'] = best_val_metrics
    return results