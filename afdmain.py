from flearn.experiments.adapdrop import AdapDrop
from flearn.utils.options import args_parser


def main():
    args = args_parser()

    result_dir = args.result_dir
    equal = True if args.equal == 1 else False
    unequal = True if args.unequal == 1 else False
    unbalance = True if args.unbalance == 1 else False
    dirichlet = True if args.dirichlet == 1 else False
    l = args.l
    prune_rate = args.prune_rate

    t = AdapDrop(args, equal=equal, unequal=unequal, unbalance=unbalance, dirichlet=dirichlet, l=l,
                 prune_rate=prune_rate, result_dir=result_dir)
    t.train()


if __name__ == '__main__':
    main()
