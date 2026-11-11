import algorithms
import os
from utils.validate import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from utils.args import *
from utils.misc import *
from dataset.data_manager import get_dataset
from tqdm import tqdm

if __name__ == "__main__":

    args = get_args()
    cfg = setup_cfg(args)
    # log_path = os.path.join('./result/fundusaug', cfg.OUTPUT_PATH)
    log_path = cfg.OUTPUT_PATH

    # init
    train_loader, val_loader, test_loader, dataset_size = get_dataset(cfg)
    writer = init_log(args, cfg, log_path, len(train_loader), dataset_size)
    algorithm_class = algorithms.get_algorithm_class(cfg.ALGORITHM)
    algorithm = algorithm_class(cfg.DATASET.NUM_CLASSES, cfg)
    algorithm.cuda()

    # train
    iterator = tqdm(range(cfg.EPOCHS))
    scheduler = get_scheduler(algorithm.optimizer, cfg.EPOCHS)
    best_performance = 0.0
    for i in iterator:

        epoch = i + 1
        loss_avg = LossCounter()
        srom_sums = {}
        srom_steps = 0
        for image, mask, label, domain, img_index in train_loader:

            algorithm.train()
            minibatch = [image.cuda(), mask.cuda(), label.cuda().long(), domain.cuda().long()]
            loss_dict_iter = algorithm.update(minibatch)
            loss_avg.update(loss_dict_iter['loss'])
            if cfg.ALGORITHM == 'GDRNetSROM':
                srom_steps += 1
                for key, value in loss_dict_iter.items():
                    if key.startswith('loss_') or key == 'active_sae' or key.startswith('steer_'):
                        srom_sums[key] = srom_sums.get(key, 0.0) + float(value)

        if srom_steps:
            summary = {k: v / srom_steps for k, v in srom_sums.items()}
            logging.info('epoch: %s, SROM losses: %s', epoch, summary)
            for key, value in summary.items():
                writer.add_scalar('SROM/' + key, value, epoch)
        alpha = algorithm.update_epoch(epoch)
        update_writer(writer, epoch, scheduler, loss_avg)
        scheduler.step()

        # validation
        if epoch % cfg.VAL_EPOCH == 0:
            # val_auc, test_auc = algorithm.validate(val_loader, test_loader, writer)
            # if val_auc > best_performance and epoch > cfg.EPOCHS * 0.3:
            val_acc, test_acc = algorithm.validate(val_loader, test_loader, writer)
            if val_acc > best_performance and epoch > cfg.EPOCHS * 0.3:
                best_performance = val_acc
                algorithm.save_model(log_path)
            # algorithm.save_model(log_path, test_acc, epoch)

    algorithm.renew_model(log_path)
    # _, test_auc = algorithm.validate(val_loader, test_loader, writer)
    _, test_acc = algorithm.validate(val_loader, test_loader, writer)


    # os.mknod(os.path.join(log_path, 'done'))
    writer.close()
