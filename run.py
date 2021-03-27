from models.trainer import Trainer
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n_neighbors", dest="n_neighbors", help="n_neighbors", default=3, type=int)
    args = parser.parse_args()

    print(args)
    
    trainer = Trainer(args.n_neighbors)
    score = trainer.train_model()
    print(score)
