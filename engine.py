import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = "1"
if os.environ.get('DEBUG', False): print('\033[92m'+'Running code in DEBUG mode'+'\033[0m')

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import logging

from models import build_model
from processors import build_processor
from utils import set_seed
from runner.runner import Runner
import wandb

logger = logging.getLogger(__name__)


def run(args, model, processor, optimizer, scheduler):

    set_seed(args)

    logger.info("train dataloader generation")
    train_examples, train_features, train_dataloader, args.train_invalid_num = processor.generate_dataloader('train')
    logger.info("dev dataloader generation")
    dev_examples, dev_features, dev_dataloader, args.dev_invalid_num = processor.generate_dataloader('dev')
    logger.info("test dataloader generation")
    test_examples, test_features, test_dataloader, args.test_invalid_num = processor.generate_dataloader('test')

    runner = Runner(
        cfg=args,
        data_samples=[train_examples, dev_examples, test_examples],
        data_features=[train_features, dev_features, test_features],
        data_loaders=[train_dataloader, dev_dataloader, test_dataloader],
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn_dict=None,
    )
    runner.run()


def main():
    from config_parser import get_args_parser
    args = get_args_parser()

    # # '''wandb'''
    # wandb.init(project="paie-gen-new")
    # args.keep_ratio = wandb.config.sample_rate
    # args.seed = wandb.config.seed
    # args.learning_rate = wandb.config.lr
#  ——————————————————



    if not args.inference_only:
        print(f"Output full path {os.path.join(os.getcwd(), args.output_dir)}")
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logging.basicConfig(
            filename=os.path.join(args.output_dir, "log.txt"), \
            format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt='%m/%d/%Y %H:%M:%S', level = logging.INFO
            )
    else:
        logging.basicConfig(
            format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', \
            datefmt = '%m/%d/%Y %H:%M:%S', level = logging.INFO
            )
    set_seed(args)

    model, tokenizer, optimizer, scheduler = build_model(args, args.model_type)
    # model, tokenizer, optimizer, scheduler = build_model(args, args.model_type, wandb.config.strategy)
    model.to(args.device)

#  ——————————————————
#     model.set_loss_rate(wandb.config.loss_rate)
#  ——————————————————————

    processor = build_processor(args, tokenizer)

    logger.info("Training/evaluation parameters %s", args)
    run(args, model, processor, optimizer, scheduler)

if __name__ == "__main__":

    # sweep_config = {
    #     'method': 'grid'
    # }
    # metric = {
    #     'name': "related test-f1 score",
    #     'goal': 'maximize'
    # }
    # sweep_config['metric'] = metric
    # sweep_config['parameters'] = {}
    # sweep_config['parameters'].update({
    #     # 'loss_rate': {
    #     #     'values': [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    #     # },
    #     'seed': {
    #         'values': [44, 13,21,88,100]
    #     },
    #     'lr': {'values':["5e-5", "3e-5", "2e-5"]},
    #     # 'lr': {'values': ["5e-5"]},
    #     # 'strategy':{'values':[1,2,3,4,5]},
    #     'loss_rate': {
    #         'values': [0.5,0.2,0.8]
    #     },
    #     # 'loss_rate': {
    #     #     'values': [0.5]
    #     # },
    #     'sample_rate': {
    #         'values': [1.0,0.5,0.1,0]
    #     },
    #
    # })
    #
    # sweep_id = wandb.sweep(sweep_config, project="paie-gen-new")
    # wandb.agent(sweep_id, function=main)

    main()