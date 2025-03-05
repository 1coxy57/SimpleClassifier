from .train import Trainer

def main():
    t = Trainer()
    res = t.train()
    return res


if __name__ == "__main__":
    main()
