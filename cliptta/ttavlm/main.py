import os

from argparse import Namespace as ArgsType

import math
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import ttavlm.lib as lib
import ttavlm.configuration as config

from ttavlm.datasets import return_train_val_datasets, return_ood_dataset, get_template
from ttavlm.datasets import CORRUPTIONS, DOMAINS, DATASET_SUITE, VISDA_DOMAINS, PACS_DOMAINS, OFFICEHOME_DOMAINS, CLEAN_DATASETS
from ttavlm.methods import return_tta_model
from ttavlm.models import return_base_model
from ttavlm.transforms import TransformList, add_tta_transform

if os.environ.get("TORCH_DETECT_ANOMALY", "").lower() in {"1", "true", "yes"}:
    torch.autograd.set_detect_anomaly(True)


class FirstBatchLargerBatchSampler(BatchSampler):
    def __init__(self, sampler, batch_size: int, first_batch_multiplier: int, drop_last: bool) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if first_batch_multiplier < 1:
            raise ValueError(f"first_batch_multiplier must be >= 1, got {first_batch_multiplier}")
        self.sampler = sampler
        self.batch_size = int(batch_size)
        self.first_batch_size = int(batch_size) * int(first_batch_multiplier)
        self.drop_last = bool(drop_last)

    def __iter__(self):
        current_batch_size = self.first_batch_size
        batch = []
        for index in self.sampler:
            batch.append(index)
            if len(batch) == current_batch_size:
                yield batch
                batch = []
                current_batch_size = self.batch_size
        if batch and not self.drop_last:
            yield batch

    def __len__(self) -> int:
        total = len(self.sampler)
        if total <= 0:
            return 0
        if self.drop_last:
            if total < self.first_batch_size:
                return 0
            remaining = total - self.first_batch_size
            return 1 + remaining // self.batch_size
        if total <= self.first_batch_size:
            return 1
        remaining = total - self.first_batch_size
        return 1 + math.ceil(remaining / self.batch_size)


def build_main_loader(dataset, batch_size: int, workers: int, first_batch_multiplier: int) -> DataLoader:
    if first_batch_multiplier <= 1:
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            pin_memory=False,
            drop_last=False,
        )

    sampler = RandomSampler(dataset)
    batch_sampler = FirstBatchLargerBatchSampler(
        sampler=sampler,
        batch_size=batch_size,
        first_batch_multiplier=first_batch_multiplier,
        drop_last=False,
    )
    return DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        num_workers=workers,
        pin_memory=False,
    )


def main(args: ArgsType) -> None:
    # Loading GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    trigger_sync = lambda: None  # noqa F811

    results = dict()
    for dataset in args.dataset:
        # Load base model
        base_model, base_transform = return_base_model(
            name=args.base_model_name,
            device=device,
            dataset=dataset,
            path_to_weights=args.save_root,
            segments=args.segments,
        )

        _, clean_val_dataset = return_train_val_datasets(
            name=CLEAN_DATASETS[dataset],
            data_dir=args.dataroot,
            train_transform=base_transform,
            val_transform=base_transform,
        )
        args.max_iter = math.ceil(len(clean_val_dataset) / args.batch_size)
        template = get_template(dataset, args.template_type)

        # Loading TTA model
        tta_model = return_tta_model(args.adaptation, base_model, args, template, clean_val_dataset.class_names)

        results[dataset] = lib.DictAverage()
        for seed_id, seed in enumerate(args.seeds):
            # Fixing seed
            lib.fix_seed(seed)

            if seed_id == 0:
                results[dataset]["overall"] = lib.DictAverage()
            overall_acc, overall_top5, overall_auc, overall_fpr = 0, 0, 0, 0

            severity_list = args.severity if dataset not in CLEAN_DATASETS.values() else [0]
            shift_type_list = args.shift_type if dataset not in CLEAN_DATASETS.values() or dataset in ["visda", "pacs", "officehome"] else ["original"]

            for severity in severity_list:
                if seed_id == 0:
                    results[dataset][f"[{severity}] / mean"] = lib.DictAverage()
                severity_acc, severity_top5, severity_auc, severity_fpr = 0, 0, 0, 0
                for shift_type in shift_type_list:
                    corr_severity = f"[{severity}] / " + shift_type
                    if seed_id == 0:
                        results[dataset][corr_severity] = lib.DictAverage()

                    run_wandb = None

                    # Eventually reset model
                    if not args.fully_continual:
                        tta_model.reset()
                        lib.LOGGER.info("Resetting model")

                    if args.use_tta:
                        if args.adaptation == 'zero':
                            val_transform = TransformList([base_transform] + [add_tta_transform(base_transform, 224, style="zero") for _ in range(args.n_augment)])
                        else:
                            val_transform = TransformList([base_transform] + [add_tta_transform(base_transform, 224) for _ in range(args.n_augment)])

                    else:
                        val_transform = TransformList([base_transform])

                    # Load datasets
                    _, val_dataset = return_train_val_datasets(
                        name=dataset,
                        data_dir=args.dataroot,
                        train_transform=val_transform,
                        val_transform=val_transform,
                        shift=shift_type,
                        severity=severity,
                    )
                    main_loader = build_main_loader(
                        dataset=val_dataset,
                        batch_size=args.batch_size,
                        workers=args.workers,
                        first_batch_multiplier=args.initial_batch_multiplier,
                    )
                    lib.LOGGER.info(f"Loading dataloader {main_loader.dataset.__class__.__name__}")

                    if not args.closed_set:
                        ood_dataset = return_ood_dataset(
                            ood_dataset_name=args.ood_dataset,
                            data_dir=args.dataroot,
                            shift_type=shift_type,
                            severity=severity,
                            transform=val_transform,
                        )
                        ood_loader = DataLoader(
                            ood_dataset,
                            batch_size=args.ood_batch_size,
                            shuffle=True,
                            num_workers=args.workers,
                            drop_last=True,
                        )
                        lib.LOGGER.info(f"Loading dataloader {ood_loader.dataset.__class__.__name__}")

                    else:
                        ood_loader = None

                    # Test-Time Adaptation
                    acc, top5, auc, fpr, _ = tta_model.get_results(main_loader, ood_loader, run_wandb, trigger_sync, args.display_progress)

                    results[dataset][corr_severity]["acc"].update(acc)
                    results[dataset][corr_severity]["top1"].update(acc)
                    results[dataset][corr_severity]["top5"].update(top5)
                    results[dataset][corr_severity]["auc"].update(auc)
                    results[dataset][corr_severity]["fpr"].update(fpr)
                    severity_acc += acc / len(shift_type_list)
                    severity_top5 += top5 / len(shift_type_list)
                    severity_auc += auc / len(shift_type_list)
                    severity_fpr += fpr / len(shift_type_list)


                # Compute average results over all corruptions for given severity
                results[dataset][f"[{severity}] / mean"]["acc"].update(severity_acc)
                results[dataset][f"[{severity}] / mean"]["top1"].update(severity_acc)
                results[dataset][f"[{severity}] / mean"]["top5"].update(severity_top5)
                results[dataset][f"[{severity}] / mean"]["auc"].update(severity_auc)
                results[dataset][f"[{severity}] / mean"]["fpr"].update(severity_fpr)
                overall_acc += severity_acc / len(severity_list)
                overall_top5 += severity_top5 / len(severity_list)
                overall_auc += severity_auc / len(severity_list)
                overall_fpr += severity_fpr / len(severity_list)

            # Compute average results over all corruptions and severities
            results[dataset]["overall"]["acc"].update(overall_acc)
            results[dataset]["overall"]["top1"].update(overall_acc)
            results[dataset]["overall"]["top5"].update(overall_top5)
            results[dataset]["overall"]["auc"].update(overall_auc)
            results[dataset]["overall"]["fpr"].update(overall_fpr)

    lib.print_results(results)


if __name__ == "__main__":
    args = config.argparser()
    lib.setup_logger()

    if args.debug:
        os.environ["DEBUG"] = "True"
        lib.LOGGER.info("Launching TOSTTA in debug mode.")

    if args.dataset == ["dataset_suite"]:
        args.dataset = DATASET_SUITE

    if any(dataset in ["cifar10", "cifar100", "imagenet"] for dataset in args.dataset):
        args.severity = [0]
        args.shift_type = ["original"]

    if args.dataset == ["domains"]:
        args.dataset = DOMAINS

    if args.shift_type == ["all"]:
        if any(data in ["cifar10c", "cifar100c", "imagenetc"] for data in args.dataset):
            args.shift_type = CORRUPTIONS
        elif any(data in ["visda"] for data in args.dataset):
            args.shift_type = VISDA_DOMAINS
        elif any(data in ["pacs"] for data in args.dataset):
            args.shift_type = PACS_DOMAINS
        elif any(data in ["officehome"] for data in args.dataset):
            args.shift_type = OFFICEHOME_DOMAINS

    lib.LOGGER.info(
        f"Launching TOSTTA on {getattr(args, 'env', 'local')}'s environment, "
        f"the dataroot is {args.dataroot}."
    )

    main(args)
