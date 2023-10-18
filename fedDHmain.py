from flearn.experiments.fedDHresnet import FedDH
from flearn.utils.options import args_parser

def main():
    args = args_parser()

    result_dir = args.result_dir
    iid = True if args.iid == 1 else False
    equal = True if args.equal == 1 else False
    unequal = True if args.unequal == 1 else False
    unbalance = True if args.unbalance == 1 else False
    dirichlet = True if args.dirichlet ==1 else False
    l = args.l


    t = FedDH(args, iid=iid, equal=equal, unequal=unequal, unbalance=unbalance, dirichlet=dirichlet, l=l, result_dir=result_dir)
    t.train()


if __name__ == '__main__':
    main()
