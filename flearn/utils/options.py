import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=1,
                        help="number of rounds of training")
    parser.add_argument('--num_users', type=int, default=100,
                        help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.1,
                        help='the fraction of clients: C')
    parser.add_argument('--local_ep', type=int, default=3, #5
                        help="the number of local epochs: E")
    parser.add_argument('--local_bs', type=int, default=10, #10
                        help="local batch size: B")
    parser.add_argument('--lr', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--regular', type=int, default=1,
                        help="lambda regularization 1 for yes")
    parser.add_argument('--lr_lambda', type=float, default=0.1,
                        help='lambda learning rate')
    parser.add_argument('--lr_b', type=float, default=0.01,
                            help='b learning rate')
    parser.add_argument("--lambda_decay", type=float, default=0.99, help="decay for update lambda")
    parser.add_argument("--b_decay", type=float, default=0.99, help="decay for update b")
    parser.add_argument('--momentum', type=float, default=0.5,
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--global_momentum', type=int, default=0,
                        help="use global_momentum to update 1 for yes")
    parser.add_argument("--update_lambda", type=int, default=1,
                        help="Default set to update lambda. Set to 0 for non update train")

    # share arguments
    parser.add_argument('--dataset', type=str, default='tinyimagenet', help="name of dataset")
    parser.add_argument("--decay", type=float, default=0.99, help="decay for central training")

    # model arguments
    parser.add_argument('--model', type=str, default='resnet', help='model name')
    parser.add_argument('--kernel_num', type=int, default=9,
                        help='number of each kind of kernel')
    parser.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to \
                        use for convolution')
    parser.add_argument('--num_channels', type=int, default=1, help="number \
                        of channels of images")
    parser.add_argument('--norm', type=str, default='batch_norm',
                        help="batch_norm, layer_norm, or None")
    parser.add_argument('--num_filters', type=int, default=32,
                        help="number of filters for conv nets -- 32 for \
                        mini-imagenet, 64 for omiglot.")
    parser.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than \
                        strided convolutions")

    # prune arguments
    parser.add_argument("--prune_rate", type=float, default=0.6, help="the rate for pruning")
    parser.add_argument("--start_epoch", type=int, default=0, help="start dropout epoch")


    # prox agruments
    parser.add_argument("--client_mu", type=float, default=0.0, help="the client norm param")

    # nova arguments
    parser.add_argument("--gradient_norm", type=int, default=0, help="Default set to no gradient normalization. Set to 1 for gradient normalizaiton")

    # scaffold agruments
    parser.add_argument("--control", type=int, default=0, help="Default set to no scaffold. Set to 1 for server, client control.")

    # other arguments
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=None, help="To use cuda, set \
                        to a specific GPU ID. Default set to use CPU.")
    parser.add_argument('--optim', type=str, default='sgd', help="type \
                        of optim")
    parser.add_argument('--iid', type=int, default=0,
                        help='Default set to IID. Set to 0 for non-IID.')
    parser.add_argument('--equal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                            non-i.i.d setting')
    parser.add_argument('--unequal', type=int, default=0,
                        help='whether to use unequal data splits for  \
                        non-i.i.d setting')
    parser.add_argument('--l', type=int, default=2,
                        help="the category each device have")
    parser.add_argument('--unbalance', type=int, default=0,
                        help='whether to use unbalance data splits for  \
                            non-i.i.d setting')
    parser.add_argument('--dirichlet', type=int, default=1,
                        help='whether to use dirichlet distribution data splits for  \
                            non-i.i.d setting')
    parser.add_argument('--stopping_rounds', type=int, default=10,
                        help='rounds of early stopping')
    parser.add_argument('--verbose', type=int, default=0, help='verbose')
    parser.add_argument('--seed', type=int, default=777, help='random seed')
    parser.add_argument("--result_dir", type=str, help="dir name for save result", required=False)

    args = parser.parse_args()
    return args
