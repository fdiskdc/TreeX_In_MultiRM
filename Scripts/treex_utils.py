"""
Shared utility functions for TreeX training scripts.
"""
from tqdm import tqdm
import logging
from datetime import datetime
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, lr_scheduler
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score, average_precision_score
import csv
import json
import prettytable as pt
from time import time

# Import project-specific modules
# These are imported at module level to avoid "wildcard import not allowed within a function" error in Python 3.7+
try:
    from util_layers import *
    from models import *
    from train_utils import *
    from main_model import RNA_ClassQuery_Model_Treex, ParallelCNNBlock, ClassQueryHead, ClassQueryHeadPooling, HierarchicalClassQueryHeadPooling
except ImportError:
    # These will be imported when needed (after PYTHONPATH is set)
    pass


def setup_logging(base_dir='logs'):
    """Setup logging with timestamped subfolder for each experiment."""
    # Create logs directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Create timestamped subfolder
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(base_dir, timestamp)
    os.makedirs(log_dir, exist_ok=True)

    # Setup logging
    log_file = os.path.join(log_dir, 'experiment.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Also print to console
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log directory: {log_dir}")
    return logger, log_dir


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def cal_loss_weight(dataset, beta=0.99999):
    data, label = dataset[:]
    total_example = label.shape[0]
    num_task = label.shape[1]
    labels_dict = dict(zip(range(num_task),[sum(label[:,i]) for i in range(num_task)]))
    keys = labels_dict.keys()
    class_weight = dict()

    # Class-Balanced Loss Based on Effective Number of Samples
    for key in keys:
        effective_num = 1.0 - beta**labels_dict[key]
        weights = (1.0 - beta) / effective_num
        class_weight[key] = weights

    weights_sum = sum(class_weight.values())

    # normalizing weights
    for key in keys:
        class_weight[key] = class_weight[key] / weights_sum * num_task

    return class_weight


def naive_loss(y_pred, y_true, loss_weight=None, ohem=False, focal=False):
    num_task = y_true.shape[-1]
    num_examples = y_true.shape[0]
    k = 0.7

    def binary_cross_entropy(x, y, focal=False):
        alpha = 0.75
        gamma = 2

        pt = x * y + (1 - x) * (1 - y)
        at = alpha * y + (1 - alpha)* (1 - y)

        # focal loss
        if focal:
            loss = -at*(1-pt)**(gamma)*(torch.log(x) * y + torch.log(1 - x) * (1 - y))
        else:
            loss = -(torch.log(x) * y + torch.log(1 - x) * (1 - y))
        return loss

    loss_output = torch.zeros(num_examples).cuda()

    # Handle model returning tuple (logits, attn_weights) from hierarchical head
    if isinstance(y_pred, tuple) and len(y_pred) == 2:
        # Extract logits (first element), ignore attention weights
        y_pred = y_pred[0]

    # Handle both list of tensors and single tensor [Batch, Num_Classes]
    if isinstance(y_pred, (list, tuple)):
        # Old format: list of tensors
        for i in range(num_task):
            if loss_weight:
                out = loss_weight[i]*binary_cross_entropy(y_pred[i], y_true[:,i], focal)
                loss_output += out
            else:
                loss_output += binary_cross_entropy(y_pred[i], y_true[:,i], focal)
    else:
        # New format: single tensor [Batch, Num_Classes]
        for i in range(num_task):
            if loss_weight:
                out = loss_weight[i]*binary_cross_entropy(y_pred[:,i], y_true[:,i], focal)
                loss_output += out
            else:
                loss_output += binary_cross_entropy(y_pred[:,i], y_true[:,i], focal)

    # Online Hard Example Mining
    if ohem:
        val, idx = torch.topk(loss_output, int(k*num_examples))
        loss_output[loss_output<val[-1]] = 0

    loss = torch.sum(loss_output)
    return loss


def naive_loss2(y_pred, y_true, loss_weight=None, ohem=False, focal=False, eps=1e-8, k=0.7):
    """
    Optimized version of naive_loss with:
    1. Vectorized computation (eliminate for loops)
    2. Numerical stability (clamp to avoid log(0))
    3. Unified data format handling (list/tensor)
    4. Device compatibility (auto-detect device)

    Args:
        y_pred: Predictions (list, tuple of tensors, or tensor [Batch, Num_Tasks])
        y_true: Ground truth labels [Batch, Num_Tasks]
        loss_weight: Optional per-task weights (list or tensor)
        ohem: Online Hard Example Mining (default: False)
        focal: Use focal loss (default: False)
        eps: Small value for numerical stability (default: 1e-8)
        k: Keep ratio for OHEM (default: 0.7)

    Returns:
        Scalar loss value
    """
    num_examples = y_true.shape[0]
    device = y_true.device  # Auto-detect device

    # Extract logits if model returns tuple (logits, attn_weights)
    if isinstance(y_pred, tuple) and len(y_pred) == 2:
        y_pred = y_pred[0]

    # Unify format: convert list to tensor [Batch, Num_Tasks]
    if isinstance(y_pred, list):
        y_pred = torch.stack(y_pred, dim=1)

    # Ensure y_pred is on the same device as y_true
    if y_pred.device != device:
        y_pred = y_pred.to(device)

    # Numerical stability: clamp predictions to [eps, 1-eps]
    y_pred_clamped = torch.clamp(y_pred, eps, 1 - eps)

    # Compute pt and at for focal loss
    # pt: confidence score (high when prediction is correct)
    pt = y_pred_clamped * y_true + (1 - y_pred_clamped) * (1 - y_true)
    # at: alpha weighting (0.75 for positive class, 0.25 for negative)
    alpha = 0.75
    at = alpha * y_true + (1 - alpha) * (1 - y_true)

    # Base binary cross entropy
    bce = -(torch.log(y_pred_clamped) * y_true + torch.log(1 - y_pred_clamped) * (1 - y_true))

    # Apply focal loss if enabled
    if focal:
        gamma = 2
        per_task_loss = at * (1 - pt) ** gamma * bce
    else:
        per_task_loss = bce

    # Apply loss weights if provided
    if loss_weight is not None:
        # Convert list to tensor if needed (use stack to preserve requires_grad)
        if isinstance(loss_weight, list):
            loss_weight_tensor = torch.stack(loss_weight) if isinstance(loss_weight[0], torch.Tensor) else torch.tensor(loss_weight, device=device)
        else:
            loss_weight_tensor = loss_weight
        per_task_loss = per_task_loss * loss_weight_tensor

    # Sum across tasks to get per-example loss [Batch]
    loss_output = per_task_loss.sum(dim=-1)

    # Online Hard Example Mining: keep only top-k% hardest examples
    if ohem:
        kth_value = torch.kthvalue(loss_output, int(k * num_examples)).values
        loss_output = torch.where(loss_output >= kth_value, loss_output, torch.zeros_like(loss_output))

    return loss_output.sum()


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def test(model, test_loader, loss_weight, use_embedding, use_uncertain_weighting, MultiTaskLossWrapper=None):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        metrics_dict = {"acc": 0, "auc": 0, "ap": 0}
        for x, y_true in test_loader:
            x, y_true = x.cuda(), y_true.cuda()
            # resize input x from 1001 features
            if not use_embedding:
                x = x.view(x.size(0), -1, 4).transpose(1, 2)

            y_pred = model(x)
            if use_uncertain_weighting:
                MultiTaskLossWrapper.eval()
                test_loss += MultiTaskLossWrapper(y_pred, y_true)
            else:
                test_loss += naive_loss2(y_pred, y_true, loss_weight)

            # Handle model returning tuple (logits, attn_weights) from hierarchical head
            if isinstance(y_pred, tuple) and len(y_pred) == 2:
                # Extract logits (first element), ignore attention weights
                y_pred = y_pred[0]

            acc = 0
            auc = 0
            ap = 0
            # Handle both list of tensors and single tensor [Batch, Num_Classes]
            if isinstance(y_pred, (list, tuple)):
                num_task = len(y_pred)
            else:
                num_task = y_pred.shape[-1]
            for i in range(num_task):
                label = y_true.cpu().numpy()[:, i]
                if isinstance(y_pred, (list, tuple)):
                    y_score = y_pred[i].cpu().detach().numpy()
                else:
                    y_score = y_pred[:, i].cpu().detach().numpy()
                y_pred_single = np.array([0 if instance < 0.5 else 1 for instance in y_score])

                acc += np.mean(y_pred_single == label)

                try:
                    auc += roc_auc_score(label, y_score)
                except ValueError:
                    pass

                try:
                    ap += average_precision_score(label, y_score)
                except ValueError:
                    pass

            metrics_dict['acc'] += acc / num_task
            metrics_dict["auc"] += auc / num_task
            metrics_dict["ap"] += ap / num_task

        num_examples = len(test_loader.dataset)
        test_loss /= num_examples
        num_batches = num_examples // test_loader.batch_size + 1
        metrics_dict['acc'] /= num_batches
        metrics_dict['auc'] /= num_batches
        metrics_dict['ap'] /= num_batches

    return test_loss, metrics_dict


def convert_to_serializable(obj):
    """Convert numpy types to JSON-serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_metrics_to_json(metrics, save_path, logger):
    """Save metrics dictionary to JSON file."""
    serializable_metrics = {}
    for key, values in metrics.items():
        serializable_metrics[key] = [convert_to_serializable(v) for v in values]

    with open(save_path, 'w') as fp:
        json.dump(serializable_metrics, fp)
    logger.info("Storing performance results to %s" % save_path)


def log_metrics_table(metrics, class_names, logger):
    """Log metrics in a pretty table format."""
    tb = pt.PrettyTable()
    tb.field_names = ["metrics"] + class_names
    for key, values in metrics.items():
        formatted_values = []
        for v in values:
            if isinstance(v, (int, float)):
                formatted_values.append(format(v, ".2f"))
            else:
                formatted_values.append(v)
        tb.add_row([key] + formatted_values)
    logger.info('\n' + str(tb))


def log_average_metrics(metrics_avg, logger):
    """Log average metrics in a pretty table format."""
    logger.info('-'*35 + "Average Metrics" + "-"*35)
    tb_avg = pt.PrettyTable()
    tb_avg.field_names = ["metric", "value"]
    for key, value in metrics_avg.items():
        if isinstance(value, (int, float)):
            tb_avg.add_row([key, format(value, ".4f")])
        else:
            tb_avg.add_row([key, value])
    logger.info('\n' + str(tb_avg))


def train(model, train_loader, test_data, args, logger, save_dir, MultiTaskLossWrapper=None):
    """Main training loop."""
    if not os.path.exists(save_dir):
        logger.info(f'{save_dir} does not exist, create it now')
        os.mkdir(save_dir)
    logfile = open(save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile,
                 fieldnames=['epoch', 'loss', 'val_loss', 'val_acc',
                              'val_precision', 'val_recall'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(), lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay)

    best_val_acc = 0
    best_val_loss = 50
    loss_weight_ = cal_loss_weight(train_loader.dataset)

    # prepare weights parameters
    loss_weight = []
    for i in range(args.num_task):
        loss_weight.append(loss_weight_[i].clone().detach().requires_grad_(True).to("cuda"))

    # prepare uncertain weighting
    if args.use_uncertain_weighting:
        MutiTaskLoss = MultiTaskLossWrapper(args.num_task).cuda()
    else:
        MutiTaskLoss = None

    alph = 0.16  # hyperparameter of GradNorm

    # Setup AMP (Automatic Mixed Precision)
    scaler = GradScaler() if args.amp else None
    if args.amp:
        logger.info('AMP enabled for faster training')

    logger.info('Begin Training' + '-' * 70)
    for epoch in range(args.epochs):
        model.train()
        ti = time()
        training_loss = 0.0
        coef = 0

        # Create tqdm progress bar for this epoch
        pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                      desc=f'Epoch {epoch+1}/{args.epochs}')
        for i, (x, y_true) in pbar:
            x, y_true = x.cuda(), y_true.cuda()

            if not args.use_embedding:
                x = x.view(x.size(0), -1, 4).transpose(1, 2)

            optimizer.zero_grad()

            # Use AMP autocast if enabled
            if args.amp:
                with autocast():
                    y_pred = model(x)
                    if args.use_uncertain_weighting:
                        loss = MutiTaskLoss(y_pred, y_true)
                    else:
                        loss = naive_loss2(y_pred, y_true, loss_weight, args.OHEM, args.focal_loss)

                # Scale loss and backward
                scaler.scale(loss).backward()

                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                # Update weights with scaled gradients
                scaler.step(optimizer)
                scaler.update()
            else:
                y_pred = model(x)
                if args.use_uncertain_weighting:
                    loss = MutiTaskLoss(y_pred, y_true)
                else:
                    loss = naive_loss2(y_pred, y_true, loss_weight, args.OHEM, args.focal_loss)

                # gradient clipping
                clip_value = 1
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                loss.backward()
                optimizer.step()

            training_loss += loss.data

            if i == 1:
                # sanity check y_pred
                y_pred_for_print = y_pred
                if isinstance(y_pred_for_print, tuple) and len(y_pred_for_print) == 2:
                    y_pred_for_print = y_pred_for_print[0]

                if isinstance(y_pred_for_print, (list, tuple)):
                    logger.debug("Sanity Checking, at epoch%02d, iter%02d, y_pred is" % (epoch, i) +
                            str([y_pred_for_print[j][1].cpu().detach() for j in range(args.num_task)]))
                else:
                    logger.debug("Sanity Checking, at epoch%02d, iter%02d, y_pred is" % (epoch, i) +
                            str([y_pred_for_print[1, j].cpu().detach() for j in range(args.num_task)]))
                logger.info("Learning rate: %.16f" % optimizer.state_dict()['param_groups'][0]['lr'])

            # Update progress bar with current loss
            pbar.set_postfix({'loss': loss.item()})

        lr_decay.step()

        # compute validation loss and acc
        val_loss, metrics_dict = test(model, test_data, loss_weight, args.use_embedding,
                                      args.use_uncertain_weighting, MutiTaskLoss)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss.cpu().numpy() / len(train_loader.dataset),
                                val_loss=val_loss.detach().cpu().numpy(), val_acc=metrics_dict['acc'],
                                val_recall=metrics_dict["auc"],
                                val_precision=metrics_dict['ap']))

        logger.info("===>Epoch %02d: loss=%.5f, val_loss=%.4f, val_acc=%.4f,\
                val_auc=%.4f, val_ap=%.4f, time=%ds"
              % (epoch, training_loss/len(train_loader.dataset), val_loss,
                 metrics_dict["acc"], metrics_dict["auc"], metrics_dict["ap"],
                  time()-ti))

        if metrics_dict['acc'] > best_val_acc and val_loss < best_val_loss:
            best_val_acc = metrics_dict["acc"]
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_dir + '/epoch%d.pkl' % epoch)
            logger.info("best val_acc increased to %.4f" % best_val_acc)

        logger.info("Running full test after epoch %d" % epoch)
        model.eval()
        y_pred_all = []
        y_true_all = []

        with torch.no_grad():
            for x, y_true in test_data:
                x, y_true = x.cuda(), y_true.cuda()
                if not args.use_embedding:
                    x = x.view(x.size(0), -1, 4).transpose(1, 2)
                y_pred = model(x)
                # Handle model returning tuple (logits, attn_weights) from hierarchical head
                if isinstance(y_pred, tuple) and len(y_pred) == 2:
                    # Extract logits (first element), ignore attention weights
                    y_pred = y_pred[0]
                y_pred_all.append(y_pred)
                y_true_all.append(y_true)

        # Handle both list of tensors and single tensor [Batch, Num_Classes]
        if isinstance(y_pred_all[0], (list, tuple)):
            y_pred_combined = [torch.cat([batch[i] for batch in y_pred_all], dim=0) for i in range(args.num_task)]
        else:
            y_pred_combined = torch.cat(y_pred_all, dim=0)
        y_true_combined = torch.cat(y_true_all, dim=0)

        from train_utils import cal_metrics
        class_names = test_data.dataset.class_name
        metrics, metrics_avg = cal_metrics(y_pred_combined, y_true_combined, plot=False, class_names=class_names)

        log_metrics_table(metrics, class_names, logger)
        log_average_metrics(metrics_avg, logger)

        pf_save_path = "%s/epoch%d_pf.json" % (save_dir, epoch)
        save_metrics_to_json(metrics, pf_save_path, logger)

        model.train()
    logfile.close()

    torch.save(model.state_dict(), save_dir + '/trained_model_%02dseqs.pkl' % args.length)
    logger.info('Trained model saved to \'%s/trained_model.h5\'' % (save_dir))
    logger.info("Total time = %ds" % (time() - t0))
    logger.info('End Training' + '-' * 70)

    return model


def parse_args(**kwargs):
    """Parse command line arguments for training.

    Args:
        **kwargs: Override default values for specific arguments.
                  Common overrides: length, gpu, epochs, batch_size, lr, etc.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(description="Naive Network on RBP.")

    # Apply custom defaults if provided
    defaults = {
        'inputs': kwargs.get('inputs', 'data/MultiRM_data.h5'),
        'epochs': kwargs.get('epochs', 20),
        'batch_size': kwargs.get('batch_size', 256),
        'length': kwargs.get('length', 51),
        'lr': kwargs.get('lr', 0.0001),
        'lr_decay': kwargs.get('lr_decay', 0.8),
        't_max': kwargs.get('t_max', 5),
        'save_dir': kwargs.get('save_dir', 'Results'),
        'weights': kwargs.get('weights', None),
        'gpu': kwargs.get('gpu', [0]),
        'num_task': kwargs.get('num_task', 12),
        'grad_norm': kwargs.get('grad_norm', False),
        'mode': kwargs.get('mode', 'train'),
        'use_embedding': kwargs.get('use_embedding', False),
        'use_uncertain_weighting': kwargs.get('use_uncertain_weighting', False),
        'balanced_sampler': kwargs.get('balanced_sampler', False),
        'OHEM': kwargs.get('OHEM', False),
        'focal_loss': kwargs.get('focal_loss', False),
        'hmm': kwargs.get('hmm', False),
        'use_hierarchical': kwargs.get('use_hierarchical', True),
        'use_simple_pooling': kwargs.get('use_simple_pooling', False),
        'amp': kwargs.get('amp', False),
    }

    parser.add_argument('--inputs', type=str, default=defaults['inputs'])
    parser.add_argument('--epochs', type=int, default=defaults['epochs'])
    parser.add_argument('--batch_size', type=int, default=defaults['batch_size'])
    parser.add_argument('--length', type=int, default=defaults['length'],
                        help="Sequence length")
    parser.add_argument('--lr', type=float, default=defaults['lr'],
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', type=float, default=defaults['lr_decay'],
                        help="The value multiplied by lr at each epoch")
    parser.add_argument('--t_max', type=int, default=defaults['t_max'])
    parser.add_argument('--save_dir', type=str, default=defaults['save_dir'])
    parser.add_argument('-w', '--weights', type=str, default=defaults['weights'],
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--gpu', type=int, default=defaults['gpu'], nargs='+', help="used GPU")
    parser.add_argument('--num_task', type=int, default=defaults['num_task'])
    parser.add_argument('--grad_norm', type=str2bool, default=defaults['grad_norm'], nargs='?',
                         help='activate grad norm')
    parser.add_argument('--mode', type=str, default=defaults['mode'],
                        help="Set the model to train or not")
    parser.add_argument('--use_embedding', type=str2bool, default=defaults['use_embedding'], nargs='?',
                         help='activate embedding')
    parser.add_argument('--use_uncertain_weighting', type=str2bool, default=defaults['use_uncertain_weighting'], nargs='?',
                         help='activate uncertain weighting')
    parser.add_argument('--balanced_sampler', type=str2bool, default=defaults['balanced_sampler'], nargs='?')
    parser.add_argument('--OHEM', type=str2bool, default=defaults['OHEM'], nargs='?')
    parser.add_argument('--focal_loss', type=str2bool, default=defaults['focal_loss'], nargs='?')
    parser.add_argument('--hmm', type=str2bool, default=defaults['hmm'], nargs='?')
    parser.add_argument('--use_hierarchical', type=str2bool, default=defaults['use_hierarchical'], nargs='?',
                         help='Use hierarchical class query head')
    parser.add_argument('--use_simple_pooling', type=str2bool, default=defaults['use_simple_pooling'], nargs='?',
                         help='Use simple pooling class query head')
    parser.add_argument('--amp', type=str2bool, default=defaults['amp'], nargs='?',
                         help='Enable automatic mixed precision (AMP) for faster training')
    return parser.parse_args()


def run_experiment(args, base_log_dir='logs'):
    """
    Main entry point for training scripts.

    Args:
        args: Parsed arguments from argparse
        base_log_dir: Base directory for logs (e.g., 'logs51', 'logs-101', 'logs-1001')
    """
    # Setup logging
    logger, log_dir = setup_logging(base_dir=base_log_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    # Import project-specific modules if not already imported at module level
    try:
        RNA_ClassQuery_Model_Treex
    except NameError:
        from train_utils import load_RM_data, RMdata, cal_metrics
        from main_model import RNA_ClassQuery_Model_Treex, ParallelCNNBlock, ClassQueryHead, ClassQueryHeadPooling, HierarchicalClassQueryHeadPooling

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # define model
    model = RNA_ClassQuery_Model_Treex(
        cnn_hidden_dim=128,
        cnn_kernel_sizes=(1, 3, 5, 7),
        cnn_dropout=0.1,
        num_classes=args.num_task,
        lstm_hidden_dim=256,
        lstm_num_layers=3,
        lstm_dropout=0.1,
        num_attn_heads=4,
        attn_dropout=0.1,
        use_simple_pooling=args.use_simple_pooling,
        use_hierarchical=args.use_hierarchical,
        use_layer_norm=True,
        seq_len=args.length
    )
    model.to(device)

    logger.info(str(model))

    if args.weights is not None:  # init model weights with provided one
        logger.info('Loading weights from %s' % (args.weights) + '-' * 40)
        model.load_state_dict(torch.load(args.weights))

    if args.hmm == True:
        args.use_embedding = False

    # train or test
    if args.mode == 'train':
        # load data
        train_loader, test_data = load_RM_data(args.inputs, batch_size=args.batch_size,
                                                  length=args.length,
                                                  use_embedding=args.use_embedding,
                                                  balanced_sampler=args.balanced_sampler)
        train(model, train_loader, test_data, args, logger, log_dir, MultiTaskLossWrapper)
    else:
        logger.info('Loading test data' + '-' * 70)
        test_data = RMdata(data_path=args.inputs, length=args.length,
                           use_embedding=args.use_embedding, mode=args.mode)
        x_test, y_test = test_data[:]
        torch.cuda.empty_cache()

        if not args.use_embedding:
            x_test = x_test.view(x_test.size(0), -1, 4).transpose(1, 2)

        logger.info('Begin testing %s set' % (args.mode) + '-' * 70)
        model.eval()

        # handle with CUDA memory issue
        try:
            y_pred = model(x_test.cuda())
            # Handle model returning tuple (logits, attn_weights) from hierarchical head
            if isinstance(y_pred, tuple) and len(y_pred) == 2:
                # Extract logits (first element), ignore attention weights
                y_pred = y_pred[0]
        except RuntimeError:
            logger.warning('Catch RuntimeError, prepare to batch test set' + '-' * 50)
            batch_size = 1
            num_iter = x_test.shape[0] // batch_size

            x_test_tem = x_test[0:1*batch_size, ...]
            y_pred = model(x_test_tem.cuda())
            # Handle model returning tuple (logits, attn_weights) from hierarchical head
            if isinstance(y_pred, tuple) and len(y_pred) == 2:
                # Extract logits (first element), ignore attention weights
                y_pred = y_pred[0]
            for i in range(1, num_iter):
                x_test_tem = x_test[i*batch_size:(i+1)*batch_size, ...]
                y_pred_tem = model(x_test_tem.cuda())
                # Handle model returning tuple (logits, attn_weights) from hierarchical head
                if isinstance(y_pred_tem, tuple) and len(y_pred_tem) == 2:
                    # Extract logits (first element), ignore attention weights
                    y_pred_tem = y_pred_tem[0]
                # Handle both list of tensors and single tensor [Batch, Num_Classes]
                if isinstance(y_pred, (list, tuple)):
                    for j in range(args.num_task):
                        y_pred[j] = torch.cat((y_pred[j].cpu().detach(), y_pred_tem[j].cpu().detach()), dim=0)
                else:
                    y_pred = torch.cat((y_pred.cpu().detach(), y_pred_tem.cpu().detach()), dim=0)

        class_names = test_data.class_name
        model_name = 'treex_model'
        # evaluate the model
        metrics, metrics_avg = cal_metrics(y_pred, y_test, plot=True, class_names=class_names, plot_name=model_name)

        logger.info('End testing' + '-' * 70)
        logger.info('')
        logger.info('-'*35 + "Result" + "-"*35)

        # print outcome
        log_metrics_table(metrics, class_names, logger)

        # Save to logs folder instead of args.save_dir
        pf_save_path = "%s/pf.json" % log_dir
        save_metrics_to_json(metrics, pf_save_path, logger)
