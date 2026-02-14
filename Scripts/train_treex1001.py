from tqdm import tqdm
import logging
from datetime import datetime
import os

def setup_logging(base_dir='logs1001'):
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

def naive_loss(y_pred, y_true, loss_weight=None,ohem=False,focal=False):

    num_task = y_true.shape[-1]
    num_examples = y_true.shape[0]
    k = 0.7

    def binary_cross_entropy(x, y,focal=False):
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
    # loss = nn.BCELoss(reduction='sum') fail to double backwards
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
                out = loss_weight[i]*binary_cross_entropy(y_pred[i],y_true[:,i],focal)
                loss_output += out
            else:
                loss_output += binary_cross_entropy(y_pred[i],y_true[:,i],focal)
    else:
        # New format: single tensor [Batch, Num_Classes]
        for i in range(num_task):
            if loss_weight:
                out = loss_weight[i]*binary_cross_entropy(y_pred[:,i],y_true[:,i],focal)
                loss_output += out
            else:
                loss_output += binary_cross_entropy(y_pred[:,i],y_true[:,i],focal)

    # loss = nn.MultiLabelSoftMarginLoss(weight=loss_weight,reduction='sum')
    # loss_output = loss(y_pred, y_true)

    # Online Hard Example Mining
    if ohem:
        val, idx = torch.topk(loss_output,int(k*num_examples))
        loss_output[loss_output<val[-1]] = 0

    loss = torch.sum(loss_output)
    # print(loss)
    # print(loss_output)

    return loss


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def test(model,test_loader,loss_weight,use_embedding,use_uncertain_weighting,MultiTaskLossWrapper=None):

    with torch.no_grad():
        model.eval()
        test_loss = 0
        metrics_dict = {"acc":0,
                       "auc":0,
                        "ap":0}
        for x, y_true in test_loader:
            x, y_true = x.cuda(), y_true.cuda()
            # resize input x from 1001 features
            if not use_embedding:
                x = x.view(x.size(0),-1,4).transpose(1,2)

            y_pred = model(x)
            if use_uncertain_weighting:
                MultiTaskLossWrapper.eval()
                test_loss += MultiTaskLossWrapper(y_pred,y_true)
            else:
                test_loss += naive_loss(y_pred,y_true,loss_weight)

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
                label = y_true.cpu().numpy()[:,i]
                if isinstance(y_pred, (list, tuple)):
                    y_score = y_pred[i].cpu().detach().numpy()
                else:
                    y_score = y_pred[:,i].cpu().detach().numpy()
                y_pred_single = np.array([0 if instance < 0.5 else 1 for instance in y_score])

                acc += np.mean(y_pred_single==label)

                try:
                    auc += roc_auc_score(label,y_score)
                except ValueError:
                    pass

                try:
                    ap += average_precision_score(label,y_score)
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

def train(model,train_loader,test_data,args,logger,save_dir):
    """

    """
    from time import time

    import csv
    if not os.path.exists(save_dir):
        logger.info(f'{save_dir} does not exist, create it now')
        os.mkdir(save_dir)
    logfile = open(save_dir + '/log.csv', 'w')
    logwriter = csv.DictWriter(logfile,
                 fieldnames=['epoch', 'loss', 'val_loss', 'val_acc',
                              'val_precision','val_recall'])
    logwriter.writeheader()

    t0 = time()
    optimizer = Adam(model.parameters(),lr=args.lr)
    lr_decay = lr_scheduler.ExponentialLR(optimizer,gamma=args.lr_decay)

    #lr_decay = lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.t_max)
    best_val_acc = 0
    best_val_loss = 50
    loss_weight_ = cal_loss_weight(train_loader.dataset) # dictionary
    # loss_weight = torch.Tensor(list(loss_weight.values())).cuda() # pytorch tensor

    # parepare weights parameters
    loss_weight = []
    for i in range(args.num_task):
        # initialize weights
        loss_weight.append(torch.tensor(loss_weight_[i],requires_grad=True, device="cuda"))
        # loss_weight_temp = torch.tensor(loss_weight_[i])
        # loss_weight_temp = loss_weight_temp.cuda()
        # loss_weight_temp.requires_grad = True
        # loss_weight.append(loss_weight_temp)

    # parepare uncertain weighting
    if args.use_uncertain_weighting:
        MutiTaskLoss = MultiTaskLossWrapper(args.num_task).cuda()
    else:
        MutiTaskLoss = None

    # optimizer for Grad Norm
    # optimizer_2 = Adam(loss_weight,lr=0.001) deprecated grad norm

    alph = 0.16  # hyperparameter of GradNorm

    logger.info('Begin Training'+'-' * 70)
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

            # A 1 0 0 0
            # C 0 1 0 0
            # T/U 0 0 0 1
            # G 0 0 1 0
            # N 0 0 0 0

            if not args.use_embedding:
                x = x.view(x.size(0),-1,4).transpose(1,2)


            y_pred = model(x)

            #print('%d-batch'%i, loss_weight[0].is_leaf)
            if args.use_uncertain_weighting:
                loss = MutiTaskLoss(y_pred,y_true)
            else:
                loss = naive_loss(y_pred,y_true,loss_weight,args.OHEM,args.focal_loss)


            optimizer.zero_grad()

            # gradient clipping
            clip_value = 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)


            loss.backward()
            # print(loss_weight[0].is_leaf)
            optimizer.step()
            # print(loss_weight[0].is_leaf)

            training_loss += loss.data


            # print(loss_weight[0].is_leaf)

            # Renormalizing the losses weights
            # coef = args.num_task/sum([loss for loss in loss_weight])
            # loss_weight = [coef*loss for loss in loss_weight]

            if i == 1:
                #sanity check y_pred
                # Handle model returning tuple (logits, attn_weights) from hierarchical head
                y_pred_for_print = y_pred
                if isinstance(y_pred_for_print, tuple) and len(y_pred_for_print) == 2:
                    y_pred_for_print = y_pred_for_print[0]

                if isinstance(y_pred_for_print, (list, tuple)):
                    logger.debug("Sanity Checking, at epoch%02d, iter%02d, y_pred is"%(epoch,i)+
                            str([y_pred_for_print[j][1].cpu().detach() for j in range(args.num_task)]))
                else:
                    # New format: single tensor [Batch, Num_Classes]
                    logger.debug("Sanity Checking, at epoch%02d, iter%02d, y_pred is"%(epoch,i)+
                            str([y_pred_for_print[1, j].cpu().detach() for j in range(args.num_task)]))
                logger.info("Learning rate: %.16f" % optimizer.state_dict()['param_groups'][0]['lr'] )
                    # print("Gradient of weight: ", torch.autograd.grad(Lgrad,loss_weight[0]).detach().cpu())
            # print(loss_weight)

            # Update progress bar with current loss
            pbar.set_postfix({'loss': loss.item()})

        lr_decay.step()


        # adjust learning rate by hand
        # if 19 <= epoch < 39:
        #     adjust_learning_rate(optimizer,optimizer.state_dict()['param_groups'][0]['lr'] / 10)
        # elif 39 <= epoch < 59:
        #     adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 2))
        # elif 59 <= epoch < 79:
        #     adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 3))
        # elif epoch >= 79:
        #     adjust_learning_rate(optimizer, optimizer.state_dict()['param_groups'][0]['lr'] / (10 ** 4))


        #compute validation loss and acc
        val_loss, metrics_dict = test(model,test_data,loss_weight,args.use_embedding,
                                      args.use_uncertain_weighting,MutiTaskLoss)
        logwriter.writerow(dict(epoch=epoch, loss=training_loss.cpu().numpy() / len(train_loader.dataset),
                                val_loss=val_loss.detach().cpu().numpy(), val_acc=metrics_dict['acc'],
                                val_recall=metrics_dict["auc"],
                                val_precision=metrics_dict['ap']))

        logger.info("===>Epoch %02d: loss=%.5f, val_loss=%.4f, val_acc=%.4f,\
                val_auc=%.4f, val_ap=%.4f, time=%ds"
              %(epoch,training_loss/len(train_loader.dataset),val_loss,
                 metrics_dict["acc"], metrics_dict["auc"],metrics_dict["ap"],
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
                    x = x.view(x.size(0),-1,4).transpose(1,2)
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

        class_names = test_data.dataset.class_name
        metrics, metrics_avg = cal_metrics(y_pred_combined, y_true_combined, plot=False, class_names=class_names)

        import prettytable as pt
        tb = pt.PrettyTable()
        tb.field_names = ["metrics"] + class_names
        for key, values in metrics.items():
            formatted_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    formatted_values.append(format(v, ".2f"))
                else:
                    formatted_values.append(v)
            tb.add_row([key]+formatted_values)
        logger.info('\n' + str(tb))

        logger.info('-'*35+"Average Metrics"+"-"*35)
        tb_avg = pt.PrettyTable()
        tb_avg.field_names = ["metric", "value"]
        for key, value in metrics_avg.items():
            if isinstance(value, (int, float)):
                tb_avg.add_row([key, format(value, ".4f")])
            else:
                tb_avg.add_row([key, value])
        logger.info('\n' + str(tb_avg))

        pf_save_path = "%s/epoch%d_pf.json" % (save_dir, epoch)
        import json
        import numpy as np

        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_metrics = {}
        for key, values in metrics.items():
            serializable_metrics[key] = [convert_to_serializable(v) for v in values]

        with open(pf_save_path,'w') as fp:
            json.dump(serializable_metrics, fp)
        logger.info("Storing performance results to %s" % pf_save_path)

        model.train()
    logfile.close()

    torch.save(model.state_dict(), save_dir + '/trained_model_%02dseqs.pkl' % args.length)
    logger.info('Trained model saved to \'%s/trained_model.h5\'' % (save_dir))
    logger.info("Total time = %ds" % (time() - t0))
    logger.info('End Training' + '-' * 70)

    return model


if __name__ == "__main__":
    import argparse
    import os

    # Setup logging with timestamped subfolder
    logger, log_dir = setup_logging(base_dir='logs')

    # setting the hyper parameters
    parser = argparse.ArgumentParser(description="Naive Network on RBP.")
    parser.add_argument('--inputs',default='MultiRM_data.h5',type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--length', default=51,type=int)
    parser.add_argument('--lr', default=0.0001, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.8, type=float,
                        help="The value multiplied by lr at each epoch.Set a larger value for larger epochs")
    parser.add_argument('--t_max',default=5, type=int)
    parser.add_argument('--save_dir', default='Results')
    parser.add_argument('-w', '--weights', default=None,
                        help="The path of the saved weights. Should be specified when testing")
    parser.add_argument('--gpu', type=int, default=[0], nargs='+', help="used GPU")
    parser.add_argument('--num_task',default=12, type=int)
    parser.add_argument('--grad_norm',default=False, type=str2bool, nargs='?',
                         help='activate grad norm')
    parser.add_argument('--mode',default='train',
                        help="Set the model to train or not")
    parser.add_argument('--use_embedding',default=False, type=str2bool, nargs='?',
                         help='activate embedding')
    parser.add_argument('--use_uncertain_weighting',default=False, type=str2bool, nargs='?',
                         help='activate uncertain weighting')
    parser.add_argument('--balanced_sampler',default=False,type=str2bool,nargs='?')
    parser.add_argument('--OHEM',default=False,type=str2bool,nargs='?')
    parser.add_argument('--focal_loss',default=False,type=str2bool,nargs='?')
    parser.add_argument('--hmm',default=False,type=str2bool,nargs='?')
    parser.add_argument('--use_hierarchical',default=True,type=str2bool,nargs='?',
                         help='Use hierarchical class query head')
    parser.add_argument('--use_simple_pooling',default=False,type=str2bool,nargs='?',
                         help='Use simple pooling class query head')


    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)

    from util_layers import *
    from models import *
    from train_utils import *

    # Import the new model from main_model.py
    from main_model import  RNA_ClassQuery_Model_Treex, ParallelCNNBlock, ClassQueryHead, ClassQueryHeadPooling, HierarchicalClassQueryHeadPooling


    import torch
    import torch.nn as nn
    import numpy as np
    from torch.utils.data import Dataset, DataLoader
    from torch.optim import Adam, lr_scheduler

    from sklearn.metrics import roc_auc_score, average_precision_score


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    logger.info(str(model))

    if args.weights is not None:  # init model weights with provided one
        logger.info('Loading weights from %s' % (args.weights) +'-' * 40)
        model.load_state_dict(torch.load(args.weights))


    if args.hmm == True:
        args.use_embedding = False
    # train or test
    if args.mode=='train':
        # load data
        train_loader, test_data = load_RM_data(args.inputs,batch_size=args.batch_size,
                                                  length=args.length,
                                                  use_embedding=args.use_embedding,
                                                  balanced_sampler=args.balanced_sampler)
        train(model,train_loader,test_data,args,logger,log_dir)
    else:
        logger.info('Loading test data'+'-' * 70)
        test_data = RMdata(data_path=args.inputs, length=args.length,
                           use_embedding=args.use_embedding, mode=args.mode)
        x_test, y_test = test_data[:]
        torch.cuda.empty_cache()
        # x_test, y_test = x_test.cuda(), y_test.cuda()

        if not args.use_embedding:
            x_test = x_test.view(x_test.size(0),-1,4).transpose(1,2)

        logger.info('Begin testing %s set'%(args.mode)+'-' * 70)
        model.eval()

        # handle with CUDA memory issue
        try:
            y_pred = model(x_test.cuda())
            # Handle model returning tuple (logits, attn_weights) from hierarchical head
            if isinstance(y_pred, tuple) and len(y_pred) == 2:
                # Extract logits (first element), ignore attention weights
                y_pred = y_pred[0]
        except RuntimeError:
            logger.warning('Catch RuntimeError, prepare to batch test set'+ '-'* 50)
            batch_size = 1
            num_iter = x_test.shape[0] // batch_size

            x_test_tem = x_test[0:1*batch_size,...]
            y_pred = model(x_test_tem.cuda())
            # Handle model returning tuple (logits, attn_weights) from hierarchical head
            if isinstance(y_pred, tuple) and len(y_pred) == 2:
                # Extract logits (first element), ignore attention weights
                y_pred = y_pred[0]
            for i in range(1, num_iter):
                x_test_tem = x_test[i*batch_size:(i+1)*batch_size,...]
                y_pred_tem = model(x_test_tem.cuda())
                # Handle model returning tuple (logits, attn_weights) from hierarchical head
                if isinstance(y_pred_tem, tuple) and len(y_pred_tem) == 2:
                    # Extract logits (first element), ignore attention weights
                    y_pred_tem = y_pred_tem[0]
                # Handle both list of tensors and single tensor [Batch, Num_Classes]
                if isinstance(y_pred, (list, tuple)):
                    for j in range(args.num_task):
                        y_pred[j] = torch.cat((y_pred[j].cpu().detach(),y_pred_tem[j].cpu().detach()),dim=0)
                else:
                    y_pred = torch.cat((y_pred.cpu().detach(),y_pred_tem.cpu().detach()),dim=0)

        class_names = test_data.class_name
        model_name = 'treex_model'
        # evaluate the model
        metrics, metrics_avg = cal_metrics(y_pred, y_test, plot=True, class_names=class_names, plot_name=model_name)

        # metrics, metrics_avg = cal_metrics_sampling(y_pred,y_test)

        # performances_df = pd.DataFrame.from_dict(metrics)
        # performances_df['names'] = pd.Series(class_names)
        # performances_df.set_index('names',inplace=True)
        # performances_df.to_csv('./pf.csv')

        logger.info('End testing'+'-' * 70)
        logger.info('')
        logger.info('-'*35+"Result"+"-"*35)
        # print outcome
        import prettytable as pt
        tb = pt.PrettyTable()
        tb.field_names = ["metrics"] + class_names
        for key, values in metrics.items():
            formatted_values = []
            for v in values:
                if isinstance(v, (int, float)):
                    formatted_values.append(format(v, ".2f"))
                else:
                    formatted_values.append(v)
            tb.add_row([key]+formatted_values)
        logger.info('\n' + str(tb))

        # Save to logs folder instead of args.save_dir
        pf_save_path = "%s/pf.json" % log_dir
        import json
        import numpy as np

        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        serializable_metrics = {}
        for key, values in metrics.items():
            serializable_metrics[key] = [convert_to_serializable(v) for v in values]

        with open(pf_save_path,'w') as fp:
            json.dump(serializable_metrics, fp)
        logger.info("Storing performance results to %s/pf.json" % log_dir)
