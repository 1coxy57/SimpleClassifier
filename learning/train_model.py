try:
    from learning.train import *
except:
    from train import *
import asyncio

class Log:
    @staticmethod
    def epoch_over(epoch: int, metrics: dict):
        print(epoch,metrics)
    
    @staticmethod
    def train_start() -> None:
        print('train start')

    @staticmethod
    def epoch_end(results: dict):
        print(results)


async def main():
    t = Trainer()
    call = callback(
        ep_end=Log.epoch_over,
        train_start=Log.train_start,
        train_end=Log.epoch_end,

    )
    t.add_callback(call)
    res = await t.train(config="data.yaml")
    return res


if __name__ == "__main__":
    asyncio.run(main())
