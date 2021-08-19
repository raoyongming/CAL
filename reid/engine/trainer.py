import logging
import torch
import torch.nn.functional as F
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP


def create_supervised_trainer(model, optimizer, loss_fn,using_cal,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        model.to(device)
        
    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch

        img = img.cuda()
        target = target.cuda()
        if using_cal:
            score,score_hat, feat = model(img)
            loss = loss_fn(score, score_hat, feat, target)
        else:
            score,feat = model(img)
            loss = loss_fn(score, feat, target)

        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()

        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        model.to(device)

    def fliplr(img):
        inv_idx = torch.arange(img.size(3)-1,-1,-1).long().cuda()  # N x C x H x W
        img_flip = img.index_select(3,inv_idx)
        return img_flip

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.cuda()
            feat = model(data)
            data_f = fliplr(data) 
            feat_f = model(data_f)
            feat = feat + feat_f
            return feat, pids, camids

    engine = Engine(_inference)
    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS
    using_cal = cfg.MODEL.CAL


    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimizer, loss_fn, using_cal,device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model, 'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.EPOCH_COMPLETED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] \nLoss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss'], 
                                engine.state.metrics['avg_acc'], scheduler.get_lr()[0]))

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if ((engine.state.epoch % eval_period == 0) or (engine.state.epoch == epochs)) and (engine.state.epoch > 1.7*epochs):           
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))



    trainer.run(train_loader, max_epochs=epochs)
