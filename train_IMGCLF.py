from argparse import ArgumentParser
from utils_clf import log2csv, ensure_path
import numpy as np
from Task_clf import LOSO
import os
from tabulate import tabulate
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--full-run', type=int, default=1, help='If it is set as 1, you will run LOSO on the same machine.')
    parser.add_argument('--test-sub', type=int, default=0, help='If full-run is set as 0, you can use this to leave this '
                                                                'subject only. Then you can divided LOSO on different '
                                                                'machines')
    ######## Data ########
    ########## there might be redudant  parameters due to custom dataset CLF
    parser.add_argument('--dataset', type=str, default='IMGCLF')
    parser.add_argument('--subjects', type=int, default=6)
    parser.add_argument('--num-class', type=int, default=39)
    parser.add_argument('--label-type', type=str, default='MW')
    parser.add_argument('--num-chan', type=int, default=39)#### changed for clf
    parser.add_argument('--num-time', type=int, default=2000)#####? for clf
    parser.add_argument('--segment', type=int, default=4, help='segment length in seconds')
    parser.add_argument('--trial-duration', type=int, default=60, help='trial duration in seconds')
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sampling-rate', type=int, default=500)
    parser.add_argument('--data-format', type=str, default='eeg')
    ######## Training Process ########
    parser.add_argument('--random-seed', type=int, default=2023)
    parser.add_argument('--max-epoch', type=int, default=200)
    parser.add_argument('--additional-epoch', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--val-rate', type=float, default=0.2)

    parser.add_argument('--save-path', default='./save/')
    parser.add_argument('--load-path', default='./data_processed/') # change this
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--mixed-precision', type=int, default=0)
    ######## Model Parameters ########
    parser.add_argument('--model', type=str, default='Deformer')
    parser.add_argument('--graph-type', type=str, default='BL', choices=['LGG-G', 'LGG-F', 'LGG-H', 'TS', 'BL'])
    parser.add_argument('--kernel-length', type=int, default=51)
    parser.add_argument('--T', type=int, default=64)
    parser.add_argument('--AT', type=int, default=16)
    parser.add_argument('--num-layers', type=int, default=6)

    args = parser.parse_args()
    
    if args.model == 'TSception':
        assert args.graph_type == 'TS', "When using TSception, suppose to get graph_type of 'TS'," \
                                        " but get {} instead!".format(args.graph_type)
        assert args.num_chan == 16, "When using TSception, suppose to have num_chan==16," \
                                    " but get {} instead!".format(args.num_chan)

    if args.model == 'LGGNet':
        assert args.graph_type in ['LGG-G', 'LGG-F', 'LGG-H'], "When using LGGNet, suppose to get graph_type " \
                                                            "of 'LGG-X'(X=G, F, or H), but get {} " \
                                                            "instead!".format(args.graph_type)

    if args.full_run:
        sub_to_run = np.arange(args.subjects)
    else:
        sub_to_run = [args.test_sub]
    logs_name = 'logs_{}_{}'.format(args.dataset, args.model)
    
    data_to_print = [
        ['dataset', args.dataset],
        ['gpu', args.gpus]
        ['model',args.model],
        ['load_path', args.load_path],
        ['batch_size', args.batch_size],
        ['max_epoch', args.max_epoch],
        ['num_class',args.num_class]
        ]
    table = tabulate(
        data_to_print,
        headers=["Parameter", "Value"],
        tablefmt="grid",
        numalign="right",
        stralign="center",
        colalign=("center", "right")
    )
    print(table)
    sub = '6_subject_clf'
    results = LOSO(
        experiment_ID='sub{}'.format(sub), args=args, logs_name=logs_name
    )
    log_path = os.path.join(args.save_path, logs_name, 'sub{}'.format(sub))
    ensure_path(log_path)
    log2csv(os.path.join(log_path, 'result.csv'), results[0])
