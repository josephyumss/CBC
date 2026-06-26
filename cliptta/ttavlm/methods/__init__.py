from typing import List
from argparse import Namespace as ArgsType

from torch import nn

from ttavlm.methods.abstract_model import AbstractOpenSetTTAModel
from ttavlm.methods.clipartt import CLIPArTT
from ttavlm.methods.cliptta_otsu import CLIPTTA
from ttavlm.methods.cliptta_reliability import CLIPTTAReliabilityGate
from ttavlm.methods.cliptta_visual_proto import CLIPTTAVisualProto
from ttavlm.methods.cliptta_fast import CLIPTTAFast
from ttavlm.methods.cbc_contrastive import CBCContrastiveTTA
from ttavlm.methods.cliptta import CLIPTTA_Old
from ttavlm.methods.source import SourceModel
from ttavlm.methods.stamp import STAMP
from ttavlm.methods.tent import Tent, TentOracle
from ttavlm.methods.lame import Lame
from ttavlm.methods.eata import ETA
from ttavlm.methods.sar import SAR
from ttavlm.methods.calip import CALIP
from ttavlm.methods.tda import TDA
from ttavlm.methods.unient import UniEnt
from ttavlm.methods.ostta import OSTTA
from ttavlm.methods.rotta import RoTTA
from ttavlm.methods.sotta import SoTTA
from ttavlm.methods.adacontrast import AdaContrast
from ttavlm.methods.watt import Watt, WattOtsu, WattUniEnt
from ttavlm.methods.zero import Zero

from ttavlm.lib import negative_classes
from ttavlm.lib.prompts import get_text_features
from ttavlm.models.clip import CLIPTextEncoder
from ttavlm.lib import LOGGER

__all__ = [
    "AbstractOpenSetTTAModel",
    "CLIPArTT",
    "CLIPTTA",
    "CLIPTTAReliabilityGate",
    "CLIPTTAVisualProto",
    "CLIPTTAFast",
    "CBCContrastiveTTA",
    "CLIPTTA_Old",
    "OSTTA",
    "SourceModel",
    "STAMP",
    "Tent",
    "TentOracle",
    "Lame",
    "ETA",
    "SAR",
    "CALIP",
    "TDA",
    "UniEnt",
    "RoTTA",
    "SoTTA",
    "AdaContrast",
    "Watt",
    "WattOtsu",
    "WattUniEnt",
    "Zero",
    "return_tta_model",
]


def return_tta_model(
    model_type: str,
    base_model: nn.Module,
    args: ArgsType,
    template: List[str] = ["a photo of a {}"],
    class_names: List[str] = None,
) -> AbstractOpenSetTTAModel:
    LOGGER.info(f"Loading {model_type}")
    LOGGER.info(f"Using template {template[0]}")
    text_names_file = getattr(args, "text_trainable_names_file", None)
    resolved_update_text = args.update_text or args.text_trainable_scope not in ["legacy", "none"] or text_names_file is not None
    if args.base_model_name.startswith("clip"):
        class_prototypes, class_bias = get_text_features(class_names, template, base_model)

        if args.score_type == "neglabel":
            negative_prototypes, negative_bias = get_text_features(negative_classes[args.dataset], template, base_model)
        else:
            negative_prototypes = None
            negative_bias = None

        clip_text_encoder = CLIPTextEncoder(base_model)
        base_model = base_model.visual
        base_model.dtype = clip_text_encoder.dtype
        normalize_features = True
        base_model.use_local = model_type == "calip"  # If using CALIP, return local and global features (otherwise only global)
    else:
        clip_text_encoder = None
        class_prototypes = base_model.fc.weight
        class_bias = base_model.fc.bias
        negative_prototypes = None
        negative_bias = None
        normalize_features = False

    base_tta_kwargs = {
        "save_root": args.save_root,
        "adaptation": args.adaptation,
        "model": base_model.eval() if model_type == "source" else base_model,
        "clip_text_encoder": clip_text_encoder,
        "class_prototypes": class_prototypes,
        "class_bias": class_bias,
        "normalize_features": normalize_features,
        "update_text": resolved_update_text,
        "update_all_params": args.update_all_params,
        "vision_trainable_scope": args.vision_trainable_scope,
        "text_trainable_scope": args.text_trainable_scope,
        "vision_trainable_names_file": getattr(args, "vision_trainable_names_file", None),
        "vision_non_ln_lr": getattr(args, "vision_non_ln_lr", None),
        "text_trainable_names_file": text_names_file,
        "optimizer_type": args.optimizer_type,
        "steps": 1 if model_type in [
            "source",
            "lame",
            "calip",
            "tda",
            "stamp",
            "adacontrast",
            "watt",
            "watt_otsu",
            "watt_unient",
            "zero",
            "cliptta_fast",
            "cbc_contrastive",
        ] else args.steps,
        "episodic": args.episodic,
        "logit_scale": args.logit_scale,
        "id_score_type": args.id_score_type,
        "ood_logit_scale": args.ood_logit_scale,
        "use_ood_loss": args.use_ood_loss,
        "detect_ood": args.detect_ood,
        "score_type": args.score_type,
        "loss_ood_type": args.loss_ood_type,
        "use_weights": args.use_weights,
        "update_alpha": args.update_alpha,
        "update_alpha_miss": args.update_alpha_miss,
        "alpha": args.alpha,
        "beta_tta": args.beta_tta,
        "beta_reg": args.beta_reg,
        "beta_ood": args.beta_ood,
        "gamma": args.gamma,
        "beta_schedule": args.beta_schedule,
        "milestone": args.milestone,
        "lr": args.lr,
        "lr_miss": args.lr_miss,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "use_sam": args.use_sam,
        "skip_top_layers": args.skip_top_layers,
        "max_iter": args.max_iter,
        "use_batch_stats_only": args.use_batch_stats_only,
        "negative_prototypes": negative_prototypes,
        "negative_bias": negative_bias,
        "distributed": args.distributed,
        "sample_size": args.sample_size,
        "num_shots": args.num_shots,
        "eval_max_batches": getattr(args, "eval_max_batches", None),
        "tsne": args.tsne,
        "measure_collapse": args.measure_collapse,
        "measure_improvement": args.measure_improvement,
    }
    if model_type == "source":
        model = SourceModel(**base_tta_kwargs)
    elif model_type == "tent":
        model = Tent(**base_tta_kwargs)
    elif model_type == "tent_oracle":
        model = TentOracle(
            oracle_miss=args.oracle_miss,
            oracle_ood=args.oracle_ood,
            miss_weight=args.miss_weight,
            **base_tta_kwargs,
        )
    elif model_type == "unient":
        model = UniEnt(
            use_cliptta_loss=args.use_cliptta_loss,
            use_clipartt_loss=args.use_clipartt_loss,
            template=template,
            class_names=class_names,
            K=args.K,
            clipartt_temp=args.clipartt_temp,
            use_memory=args.use_memory,
            **base_tta_kwargs,
        )
    elif model_type == "lame":
        model = Lame(
            affinity=args.affinity,
            **base_tta_kwargs,
        )
    elif model_type == "eta":
        model = ETA(
            d_margin=args.d_margin,
            alpha_entropy=args.alpha_entropy,
            **base_tta_kwargs,
        )
    elif model_type == "sar":
        model = SAR(
            reset_constant_em=args.reset_constant_em,
            alpha_entropy=args.alpha_entropy,
            **base_tta_kwargs,
        )
    elif model_type == "ostta":
        model = OSTTA(
            margin=args.margin_ostta,
            **base_tta_kwargs,
        )
    elif model_type == "clipartt":
        model = CLIPArTT(
            class_names=class_names,
            template=template,
            temp=args.clipartt_temp,
            K=args.K,
            **base_tta_kwargs,
        )
    elif model_type == "cliptta":
        model = CLIPTTA(
            template=template,
            class_names=class_names,
            use_softmax_entropy=args.use_softmax_entropy,
            use_memory=args.use_memory,
            use_scheduler=args.use_scheduler,
            use_tent=args.use_tent,
            use_clipartt=args.use_clipartt_loss,
            cliptta_target_mode=getattr(args, "cliptta_target_mode", "instance"),
            cliptta_instance_weight=getattr(args, "cliptta_instance_weight", 1.0),
            cliptta_classwise_weight=getattr(args, "cliptta_classwise_weight", 1.0),
            cliptta_sameclass_vv_weight=getattr(args, "cliptta_sameclass_vv_weight", 0.1),
            cliptta_proto_sym_weight=getattr(args, "cliptta_proto_sym_weight", 0.25),
            cliptta_proto_sym_agreement_only=getattr(args, "cliptta_proto_sym_agreement_only", False),
            use_relation_consistency=getattr(args, "use_relation_consistency", False),
            beta_relation=getattr(args, "beta_relation", 1.0),
            relation_temperature=getattr(args, "relation_temperature", 0.2),
            K=args.K,
            clipartt_temp=args.clipartt_temp,
            **base_tta_kwargs,
        )
    elif model_type == "cliptta_reliability":
        model = CLIPTTAReliabilityGate(
            template=template,
            class_names=class_names,
            use_softmax_entropy=args.use_softmax_entropy,
            use_memory=args.use_memory,
            use_scheduler=args.use_scheduler,
            use_tent=args.use_tent,
            use_clipartt=args.use_clipartt_loss,
            use_relation_consistency=getattr(args, "use_relation_consistency", False),
            beta_relation=getattr(args, "beta_relation", 1.0),
            relation_temperature=getattr(args, "relation_temperature", 0.2),
            K=args.K,
            clipartt_temp=args.clipartt_temp,
            vision_gate_manifest_file=getattr(args, "vision_gate_manifest_file", None),
            reliability_gate_enable=getattr(args, "reliability_gate_enable", False),
            reliability_gate_low_threshold=getattr(args, "reliability_gate_low_threshold", 0.45),
            reliability_gate_high_threshold=getattr(args, "reliability_gate_high_threshold", 0.65),
            reliability_gate_diversity_threshold=getattr(args, "reliability_gate_diversity_threshold", 0.90),
            reliability_entropy_weight=getattr(args, "reliability_entropy_weight", 0.5),
            reliability_margin_weight=getattr(args, "reliability_margin_weight", 0.5),
            **base_tta_kwargs,
        )
    elif model_type == "cliptta_visual_proto":
        model = CLIPTTAVisualProto(
            template=template,
            class_names=class_names,
            use_softmax_entropy=args.use_softmax_entropy,
            use_memory=args.use_memory,
            use_scheduler=args.use_scheduler,
            use_tent=args.use_tent,
            use_clipartt=args.use_clipartt_loss,
            use_relation_consistency=getattr(args, "use_relation_consistency", False),
            beta_relation=getattr(args, "beta_relation", 1.0),
            relation_temperature=getattr(args, "relation_temperature", 0.2),
            K=args.K,
            clipartt_temp=args.clipartt_temp,
            visual_proto_bank_size=getattr(args, "visual_proto_bank_size", 32),
            visual_proto_lambda=getattr(args, "visual_proto_lambda", 0.05),
            visual_proto_count_tau=getattr(args, "visual_proto_count_tau", 16.0),
            visual_proto_update_threshold=getattr(args, "visual_proto_update_threshold", 0.85),
            visual_proto_low_conf_threshold=getattr(args, "visual_proto_low_conf_threshold", 0.5),
            visual_proto_high_conf_threshold=getattr(args, "visual_proto_high_conf_threshold", 0.75),
            visual_proto_margin_threshold=getattr(args, "visual_proto_margin_threshold", 0.15),
            **base_tta_kwargs,
        )
    elif model_type == "cliptta_fast":
        model = CLIPTTAFast(
            template=template,
            class_names=class_names,
            fast_cache_alpha=args.fast_cache_alpha,
            fast_cache_beta=args.fast_cache_beta,
            fast_update_ratio=args.fast_update_ratio,
            fast_entropy_threshold=args.fast_entropy_threshold,
            fast_use_text_affinity=args.fast_use_text_affinity,
            **base_tta_kwargs,
        )
    elif model_type == "cbc_contrastive":
        model = CBCContrastiveTTA(
            template=template,
            class_names=class_names,
            cbc_shift_weight=getattr(args, "cbc_shift_weight", 1.0),
            cbc_contrastive_weight=getattr(args, "cbc_contrastive_weight", 1.0),
            cbc_proto_weight=getattr(args, "cbc_proto_weight", 1.0),
            cbc_diversity_weight=getattr(args, "cbc_diversity_weight", 0.2),
            cbc_target_mode=getattr(args, "cbc_target_mode", "classwise"),
            cbc_confidence_threshold=getattr(args, "cbc_confidence_threshold", 0.0),
            cbc_min_keep_ratio=getattr(args, "cbc_min_keep_ratio", 0.25),
            **base_tta_kwargs,
        )
    elif model_type == "cliptta_old":
        model = CLIPTTA_Old(
            template=template,
            class_names=class_names,
            use_softmax_entropy=args.use_softmax_entropy,
            use_memory=args.use_memory,
            use_scheduler=args.use_scheduler,
            **base_tta_kwargs,
        )
    elif model_type == "stamp":
        model = STAMP(
            memory_length=args.memory_length,
            alpha_stamp=args.alpha_stamp,
            use_consistency_filtering=args.use_consistency_filtering,
            **base_tta_kwargs,
        )
    elif model_type == "calip":
        model = CALIP(
            beta_calip=args.beta_calip,
            **base_tta_kwargs,
        )
    elif model_type == "tda":
        model = TDA(
            pos_alpha_beta=args.pos_alpha_beta,
            neg_alpha_beta=args.neg_alpha_beta,
            pos_shot_capacity=args.pos_shot_capacity,
            neg_shot_capacity=args.neg_shot_capacity,
            entropy_threshold=args.entropy_threshold,
            mask_threshold=args.mask_threshold,
            **base_tta_kwargs,
        )
    elif model_type == "rotta":
        model = RoTTA(
            capacity=args.capacity,
            update_frequency=args.update_frequency,
            lambda_u=args.lambda_u,
            lambda_t=args.lambda_t,
            alpha_rotta=args.alpha_rotta,
            nu=args.nu,
            use_tta=args.use_tta,
            **base_tta_kwargs,
        )
    elif model_type == "sotta":
        model = SoTTA(
            capacity=args.capacity,
            high_threshold=args.high_threshold,
            **base_tta_kwargs,
        )
    elif model_type == "adacontrast":
        model = AdaContrast(
            beta_ins=args.beta_ins,
            aug_type=args.aug_type,
            queue_size=args.queue_size,
            n_neighbors=args.n_neighbors,
            m=args.m,
            T_moco=args.T_moco,
            **base_tta_kwargs,
        )
    elif model_type == "watt":
        model = Watt(
            class_names=class_names,
            template=template,
            avg_type=args.avg_type,
            reps=args.reps,
            meta_reps=args.meta_reps,
            **base_tta_kwargs,
        )
    elif model_type == "watt_otsu":
        model = WattOtsu(
            class_names=class_names,
            template=template,
            avg_type=args.avg_type,
            reps=args.reps,
            meta_reps=args.meta_reps,
            **base_tta_kwargs,
        )
    elif model_type == "watt_unient":
        model = WattUniEnt(
            class_names=class_names,
            template=template,
            avg_type=args.avg_type,
            reps=args.reps,
            meta_reps=args.meta_reps,
            **base_tta_kwargs,
        )
    elif model_type == "zero":
        model = Zero(
            zero_gamma=args.zero_gamma,
            **base_tta_kwargs,
        )

    else:
        raise NotImplementedError

    return model
