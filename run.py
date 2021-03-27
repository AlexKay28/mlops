from models.trainer import Trainer

def main():
    trainer = Trainer()
    score = trainer.train_model()
    print(score)

if __name__ == "__main__":
    main()
