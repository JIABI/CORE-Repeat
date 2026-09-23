"""Two-stage training, validation-selected checkpoints, and reproducible reload."""
from __future__ import annotations

from dataclasses import asdict
from collections.abc import Mapping
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .config import TrainConfig
from .data import (TrainScaler, make_inference_batch, make_episode, collate_episodes,
                   fit_library_context, attach_library_context, LibraryBank)
from .model import MeasurementWorldModel
from .splits import assert_disjoint_splits
from .provenance import bind_fitting_provenance
from .probabilistic_scores import derived_utility_crps


def seed_everything(seed: int, threads: int = 4):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    # Deterministic CPU kernels where available. Never silently substitute an
    # unimplemented accelerator operation; this implementation targets CPU.
    torch.use_deterministic_algorithms(True)


def log_event(path, event, **values):
    row = {"utc": datetime.now(timezone.utc).isoformat(), "event": event, **values}
    line = json.dumps(row, allow_nan=False)
    with Path(path).open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


@contextmanager
def objective_random_stream(config, epoch, minibatch, *, auxiliary=False):
    """Match minibatch dropout without allowing extra draws to shift later tasks.

Legacy training remains unchanged unless paired_objective_rng is declared.
Auxiliary utility training always has its own CPU RNG scope. Episode sampling
uses its separate NumPy generator and never consumes these streams.
"""
    if not config.paired_objective_rng and not auxiliary:
        yield
        return
    seed=(config.seed+1000003*(epoch+1)+1009*(minibatch+1)
          +(1000000007 if auxiliary else 0)) % (2**63-1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        yield


def _jepa_binding(dataset, train, scaler, config):
    invariant=asdict(config)
    for key in ("objective","utility_crps_weight","utility_crps_samples","paired_objective_rng"):
        invariant.pop(key)
    digest=hashlib.sha256()
    # Bind measured train inputs/targets, acquisition metadata, and the actual
    # control catalog, so equal compound IDs cannot mask different JEPA data.
    for name in ("Y","cond","reference","reference_mask","groups","chem","well_mask","observed_mask",
                 "panel_index","panel_y","panel_ids","panel_identity","panel_members"):
        value=getattr(dataset,name,None)
        if value is None:
            continue
        array=np.asarray(value)
        if name in {"Y","cond","reference","reference_mask","groups","chem","well_mask","observed_mask","panel_index"}:
            array=array[np.asarray(train)]
        digest.update(name.encode());digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(json.dumps(array.tolist(),sort_keys=True).encode() if array.dtype.hasobject else array.tobytes())
    return {"schema_version":1,"train_ids":dataset.ids[train].tolist(),
            "feature_names":dataset.feature_names.tolist(),"scaler":scaler.to_dict(),
            "invariant_train_config":invariant,"training_measurements_digest":digest.hexdigest()}


def _load_shared_jepa(model, source, destination, binding, kwargs):
    source=Path(source)
    payload=torch.load(source/"jepa.pt",map_location="cpu",weights_only=True)
    def normalized_binding(value):
        if not isinstance(value, dict) or not isinstance(value.get("invariant_train_config"), dict):
            return value
        result = dict(value)
        invariant = dict(result["invariant_train_config"])
        defaults = asdict(TrainConfig())
        # Earlier artifacts predate these optional fields. Absence means the
        # unchanged legacy defaults, never permission to ignore a new setting.
        for name in ("biology_kernel_mode", "biology_kernel_anchors", "observation_family",
                     "lr_schedule", "warmup_steps", "min_learning_rate", "diagnostic_interval"):
            invariant.setdefault(name, defaults[name])
        result["invariant_train_config"] = invariant
        return result
    if normalized_binding(payload.get("binding")) != normalized_binding(binding) or payload.get("model_config") != kwargs:
        raise ValueError("Shared JEPA training data, scaler, geometry or invariant configuration differs")
    if "encoder_state_dict" not in payload or "post_jepa_torch_rng" not in payload:
        raise ValueError("Shared JEPA artifact lacks encoder or post-training RNG state")
    model.profile_encoder.load_state_dict(payload["encoder_state_dict"],strict=True)
    model.profile_encoder.eval().requires_grad_(False)
    torch.set_rng_state(payload["post_jepa_torch_rng"])
    shutil.copy2(source/"jepa.pt",Path(destination)/"jepa.pt")
    log_event(Path(destination)/"training.jsonl","jepa_reused",source=str(source.resolve()),
              epochs=payload["epochs"],train_compounds=len(binding["train_ids"]))


def assert_artifact_geometry(model, scaler, payload):
    names=payload.get("feature_names",payload.get("model_config",{}).get("feature_names"))
    if names is None or list(names)!=list(scaler.feature_names):
        raise ValueError("Checkpoint and scaler feature coordinates differ")
    for name,expected in (("outcome_center",scaler.y_center),("outcome_scale",scaler.y_scale)):
        recorded=getattr(model,name)
        expected=torch.as_tensor(expected,dtype=recorded.dtype,device=recorded.device)
        if recorded.shape!=expected.shape or not torch.equal(recorded,expected):
            raise ValueError("Checkpoint and scaler outcome transforms differ")


def validate_pretrained_encoder_payload(encoder, payload, *, feature_names,
                                       input_center, input_scale, protected_ids=()):
    """Validate converted profile-encoder weights before mutating an encoder.

    This is an exact-architecture import, not an adapter for arbitrary public
    image models. Identity lists must be explicitly mapped into the receiving
    dataset's compound-ID namespace. Their completeness is a provenance claim
    by the producer, not something that a tensor checkpoint can prove.
    """
    required = {"schema_version", "encoder_state_dict", "feature_names", "feature_groups",
                "encoder_config", "input_transform", "pretraining_ids", "provenance"}
    if not isinstance(payload, Mapping) or not required.issubset(payload):
        raise ValueError("Pretrained encoder requires complete weights, geometry and training provenance")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise ValueError("Unsupported pretrained encoder schema version")
    names = payload["feature_names"]
    if not isinstance(names, (list, tuple)) or list(names) != list(feature_names):
        raise ValueError("Pretrained encoder feature coordinates differ")
    groups = payload["feature_groups"]
    if not isinstance(groups, Mapping) or list(groups) != encoder.group_names:
        raise ValueError("Pretrained encoder ordered feature groups differ")
    for name, expected in encoder.feature_groups.items():
        recorded = groups[name]
        if (not isinstance(recorded, (list, tuple)) or
                any(type(i) is not int for i in recorded) or list(recorded) != expected):
            raise ValueError("Pretrained encoder feature group coordinates differ")
    architecture = payload["encoder_config"]
    expected_architecture = {"hidden_dim": encoder.hidden_dim,
                             "attention_layers": encoder.attention_layers,
                             "attention_heads": encoder.attention_heads}
    if (not isinstance(architecture, Mapping) or
            any(type(architecture.get(k)) is not int or architecture.get(k) != v
                for k, v in expected_architecture.items())):
        raise ValueError("Pretrained encoder architecture differs")
    transform = payload["input_transform"]
    if not isinstance(transform, Mapping):
        raise ValueError("Pretrained encoder input transform is missing")
    for name, expected in (("center", input_center), ("scale", input_scale)):
        try:
            recorded = np.asarray(transform.get(name), dtype=np.float64)
            expected = np.asarray(expected, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Pretrained encoder input transform is invalid") from exc
        if (recorded.shape != (encoder.feature_dim,) or recorded.shape != expected.shape or
                not np.isfinite(recorded).all() or not np.array_equal(recorded, expected) or
                (name == "scale" and np.any(recorded <= 0))):
            raise ValueError("Pretrained encoder input transform differs; an explicit adapter is required")
    provenance = payload["provenance"]
    if (not isinstance(provenance, Mapping) or
            any(not isinstance(provenance.get(k), str) or not provenance[k].strip()
                for k in ("source", "training_data")) or
            provenance.get("identity_namespace") != "opal2_dataset_compound_ids" or
            provenance.get("pretraining_ids_complete") is not True):
        raise ValueError("Pretrained encoder provenance requires source, training data and complete mapped identities")
    identities = payload["pretraining_ids"]
    if (not isinstance(identities, (list, tuple)) or not identities or
            any(not isinstance(i, str) or not i.strip() or i != i.strip() for i in identities) or
            len(set(identities)) != len(identities)):
        raise ValueError("Pretraining IDs must be a nonempty unique list of mapped compound identifiers")
    if set(map(str, protected_ids)) & set(identities):
        raise ValueError("Pretraining overlaps held-out outcome identities")
    source_groups = payload.get("pretraining_source_groups")
    if source_groups is not None:
        if (not isinstance(source_groups, (list, tuple)) or not source_groups or
                any(type(i) is not int or i < 0 for i in source_groups) or
                len(set(source_groups)) != len(source_groups) or
                provenance.get("source_namespace") != "opal2_dataset_source_groups"):
            raise ValueError("Pretraining source groups require complete mapped source provenance")
    state, expected_state = payload["encoder_state_dict"], encoder.state_dict()
    if not isinstance(state, Mapping) or set(state) != set(expected_state):
        raise ValueError("Pretrained encoder state keys differ")
    for name, expected in expected_state.items():
        value = state[name]
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided or
                value.shape != expected.shape or value.dtype != expected.dtype or
                not torch.isfinite(value).all()):
            raise ValueError(f"Pretrained encoder state tensor is incompatible: {name}")
        # Buffers are coordinate selectors, not learnable parameters. Strict
        # loading alone would silently replace them with shape-matched indices.
        if name.startswith("indices_"):
            index = int(name.removeprefix("indices_"))
            coordinates = torch.tensor(encoder.feature_groups[encoder.group_names[index]], dtype=torch.long)
            if not torch.equal(value.cpu(), coordinates):
                raise ValueError("Pretrained encoder coordinate index buffers differ")
    return {"schema_version": 1, "feature_names": list(names),
            "feature_groups": {k: list(v) for k, v in groups.items()},
            "encoder_config": expected_architecture,
            "input_transform": {k: np.asarray(transform[k], dtype=float).tolist() for k in ("center", "scale")},
            "pretraining_ids": list(identities),
            "provenance": {k: provenance[k] for k in ("source", "training_data", "identity_namespace", "pretraining_ids_complete")},
            **({"pretraining_source_groups": list(source_groups),
                 "provenance": {k: provenance[k] for k in ("source", "training_data", "identity_namespace", "pretraining_ids_complete", "source_namespace")}}
               if source_groups is not None else {})}


def load_pretrained_encoder(encoder, checkpoint, **geometry):
    """Load actual serialized compatible weights only after all checks pass."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = validate_pretrained_encoder_payload(encoder, payload, **geometry)
    encoder.load_state_dict(payload["encoder_state_dict"], strict=True)
    return metadata


def model_kwargs(dataset, config, *, chemical_anchor_data=None):
    d = dataset.dimensions
    groups = dataset.feature_groups
    if config.group_assignment == "shuffled":
        # Matched group cardinalities and total coordinates; no outcome fitting.
        # Both JEPA and the decoder consume this exact predeclared permutation.
        order = np.random.default_rng(config.seed + 1709).permutation(d["D"])
        names = list(groups)
        sizes = np.cumsum([0] + [len(groups[name]) for name in names])
        groups = {name: order[sizes[i]:sizes[i+1]].tolist() for i,name in enumerate(names)}
    kwargs = dict(feature_groups=groups, condition_dim=d["K"],
                reference_dim=d["R"], chemical_dim=d["H"], hidden_dim=config.hidden_dim,
                latent_rank=config.latent_rank, residual_rank=config.residual_rank,
                kernel_mode=config.kernel_mode,
                group_attention_layers=config.group_attention_layers,
                attention_heads=config.attention_heads,
                reference_loss_weight=config.reference_loss_weight,
                latent_kl_weight=config.latent_kl_weight,
                chemical_regularization_weight=config.chemical_regularization_weight,
                use_chemistry=config.use_chemistry, use_references=config.use_references,
                use_library=config.use_library)
    if config.observation_family != "gaussian":
        kwargs["observation_family"] = config.observation_family
    if config.biology_kernel_mode != "off":
        if chemical_anchor_data is None:
            raise ValueError("Fit chemical response anchors on training structures before constructing the model")
        kwargs.update(biology_kernel_mode=config.biology_kernel_mode,
                      chemical_anchor_data=chemical_anchor_data)
    elif chemical_anchor_data is not None:
        raise ValueError("Disabled response kernel cannot acquire chemical anchors")
    if config.use_biology_prior:
        if dataset.biology_vocabulary is None:
            raise ValueError("Fit the biological vocabulary on training units before constructing the model")
        if dataset.biology_vocabulary["evidence_weight_policy"] != config.biology_evidence_weight_policy:
            raise ValueError("Biological vocabulary and configured support-weight policies differ")
        kwargs.update(use_biology_prior=True, biology_vocabulary=dataset.biology_vocabulary)
    return kwargs


def fit_training_chemical_anchors(dataset, train, config):
    """Fit a chemistry-only basis; no Y, role, reference or risk labels are used."""
    from .biology_kernel import fit_chemical_anchors
    metadata = dict(dataset.metadata.get("chemical", {}))
    if "bits" in metadata:
        bits = metadata["bits"]
        metadata["fingerprint_indices"] = list(range(bits))
        metadata["validity_index"] = (bits if metadata.get("final_coordinate") == "valid_SMILES_indicator" else None)
    return fit_chemical_anchors(dataset.chem, dataset.chem_mask, dataset.ids,
                                dataset.ids[train].tolist(), metadata or None,
                                max_anchors=config.biology_kernel_anchors)


def make_world_model_scheduler(optimizer, config, steps_per_epoch):
    """Step-count cosine scheduling is optional; legacy plateau runs are unchanged."""
    if config.lr_schedule == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=.5, patience=5)
    total = config.epochs * steps_per_epoch
    if config.warmup_steps >= total:
        raise ValueError("Cosine warmup must leave at least one decay step")
    floor = config.min_learning_rate / config.learning_rate
    def multiplier(step):
        if step < config.warmup_steps:
            return max(floor, (step + 1) / max(1, config.warmup_steps))
        progress = min(1., (step - config.warmup_steps) / max(1, total - config.warmup_steps - 1))
        return floor + (1. - floor) * .5 * (1. + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


def batches(indices, batch_size):
    for start in range(0, len(indices), batch_size):
        yield np.asarray(indices[start:start + batch_size], dtype=int)


def apply_information_ablation(inputs, config):
    """Apply declared information removals to BOTH JEPA and probability stages."""
    out=dict(inputs)
    if not config.use_chemistry:
        out["chem"]=torch.zeros_like(out["chem"])
        out["chem_mask"]=torch.zeros_like(out["chem_mask"])
    if not config.use_references:
        for key,value in inputs.items():
            if "reference" in key or "panel" in key:
                out[key]=torch.zeros_like(value)
        for prefix in ("context","target"):
            if prefix+"_panel_index" in out:
                out[prefix+"_panel_index"]=torch.full_like(out[prefix+"_panel_index"],-1)
    if not config.use_library:
        for key,value in inputs.items():
            if key.startswith("library_"):
                out[key]=torch.full_like(value,-1) if key in {"library_index","library_group"} else torch.zeros_like(value)
    return out


def random_training_batch(dataset, indices, rng, config):
    """One freshly sampled task per compound, never across a split boundary.

    Half of states contain one observed well; the remainder uniformly choose
    feasible larger sizes. The target is every other available repetition.
    """
    episodes = []
    for i in indices:
        available = np.flatnonzero(dataset.observed_mask[i] & dataset.well_mask[i])
        if len(available) < 2:
            raise ValueError("Training compound has no distinct target repetition")
        maximum = min(4, len(available) - 1)
        size = 1 if maximum == 1 or rng.random() < .5 else int(rng.integers(2, maximum + 1))
        if rng.random() < config.zero_context_fraction:
            size = 0
        order = rng.permutation(available)
        context, target = np.sort(order[:size]), np.sort(order[size:])
        episodes.append(make_episode(dataset, int(i), context, target,
                                     reference_access=config.reference_access))
    out = collate_episodes(episodes)
    out["inputs"] = apply_information_ablation({k: v.to(config.device) for k, v in out["inputs"].items()}, config)
    out["target_y"] = out["target_y"].to(config.device)
    out["target_mask"] = out["target_mask"].to(config.device)
    return out


def fixed_batch(dataset, indices, config, contexts=(0,), targets=(1, 2, 3)):
    inputs = make_inference_batch(dataset, indices, contexts, targets,
                                  reference_access=config.reference_access, device=config.device)
    inputs=apply_information_ablation(inputs,config)
    y = dataset.Y[np.asarray(indices)][:, np.asarray(targets)]
    observed = dataset.observed_mask[np.asarray(indices)][:, np.asarray(targets)]
    y = np.where(observed[..., None], y, 0)
    return inputs, torch.as_tensor(y, dtype=torch.float32, device=config.device), torch.as_tensor(observed, device=config.device)


@torch.no_grad()
def validation_loss(model, dataset, indices, config):
    model.eval()
    total = 0.0
    for ix in batches(indices, config.batch_size):
        inputs, y, mask = fixed_batch(dataset, ix, config)
        # Select checkpoints by proper predictive marginal score. Auxiliary
        # reconstruction and variational regularization are training objectives.
        distribution = model(inputs)
        count = distribution.observed_mask(y, mask).sum()
        if count == 0:
            raise ValueError("Validation has no observable target coordinates")
        loss = -distribution.joint_log_prob(y, mask) / count
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite validation likelihood")
        total += float(loss) * len(ix)
    return total / len(indices)


def fit_jepa(dataset, train_indices, config, directory, kwargs):
    """Real cross-repeat representation training, then a frozen target encoder."""
    from .jepa import ConditionalJEPA
    from .model import GroupedProfileEncoder
    # Matching JEPA ablations start from identical representation parameters,
    # independent of how many parameters their downstream operator has.
    seed_everything(config.seed + 101, config.threads)
    directory = Path(directory)
    encoder = GroupedProfileEncoder(kwargs["feature_groups"], config.hidden_dim,
                                    attention_layers=config.group_attention_layers,
                                    attention_heads=config.attention_heads)
    learner = ConditionalJEPA(encoder, condition_dim=kwargs["condition_dim"],
                              reference_dim=kwargs["reference_dim"],
                              chemical_dim=kwargs["chemical_dim"], hidden_dim=config.hidden_dim).to(config.device)
    optimizer = torch.optim.AdamW([p for p in learner.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    rng = np.random.default_rng(config.seed + 17)
    progress = directory / "training.jsonl"
    for epoch in range(config.jepa_epochs):
        learner.train()
        sums, count = {}, 0
        for ix in batches(rng.permutation(train_indices), config.batch_size):
            item = random_training_batch(dataset, ix, rng, config)
            optimizer.zero_grad(set_to_none=True)
            result = learner.loss(item["inputs"], item["target_y"], item["target_mask"])
            loss = result["loss"]
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite JEPA training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(learner.parameters(), config.gradient_clip,
                                                 error_if_nonfinite=True)
            optimizer.step()
            learner.update_teacher()
            for k, value in result.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    sums[k] = sums.get(k, 0.0) + float(value.detach()) * len(ix)
            count += len(ix)
        log_event(progress, "jepa_epoch", epoch=epoch+1, **{k: v/count for k, v in sums.items()})
    torch.save({"state_dict": learner.state_dict(), "model_config": kwargs,
                "train_config": asdict(config), "epochs": config.jepa_epochs,
                "encoder_state_dict":learner.teacher_encoder.state_dict(),
                "post_jepa_torch_rng":torch.get_rng_state()}, directory / "jepa.pt")
    return learner.frozen_encoder()


def fit_model(dataset, splits, config: TrainConfig, directory, *, resume=False,
              shared_jepa_directory=None):
    """Fit only train, select epochs only on validation; calibration/eval untouched."""
    config.validate()
    if shared_jepa_directory is not None and resume:
        raise ValueError("Resume restores its own encoder; do not also request shared JEPA")
    if shared_jepa_directory is not None and config.resolved_encoder_policy not in {"jepa_frozen","jepa_finetune"}:
        raise ValueError("Shared JEPA requires a JEPA encoder policy")
    if config.device != "cpu":
        raise ValueError("Use CPU: exact float64 low-rank likelihood is not supported on MPS")
    assert_disjoint_splits(dataset.ids, splits)
    train, val = splits["train"], splits["validation"]
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=resume)
    progress = directory / "training.jsonl"
    seed_everything(config.seed, config.threads)
    if resume:
        payload = torch.load(directory / "last.pt", map_location="cpu", weights_only=True)
        if asdict(TrainConfig(**payload["train_config"]).validate()) != asdict(config):
            raise ValueError("Resume configuration differs from the recorded full configuration")
        if payload["train_ids"] != dataset.ids[train].tolist() or payload["validation_ids"] != dataset.ids[val].tolist():
            raise ValueError("Resume data split differs")
        scaler = TrainScaler.load(directory / "scaler.json")
    else:
        config.save(directory / "config.json")
        scaler = TrainScaler.fit(dataset, train, biology_enabled=config.use_biology_prior,
                                 biology_evidence_weight_policy=config.biology_evidence_weight_policy)
        scaler.save(directory / "scaler.json")
    normalized = scaler.transform(dataset)
    biology_binding = None
    if config.use_biology_prior:
        from .biology import BiologyRecord
        biology_records = dataset.biology_records or tuple(BiologyRecord(str(unit)) for unit in dataset.ids)
        # Checkpoint continuation must not silently acquire revised annotations.
        # Validation-only relationships never enter this predictor binding.
        biology_binding = [{"unit_id": biology_records[i].unit_id,
                            "perturbation_entities": (asdict(biology_records[i].perturbation["entities"])
                                if "entities" in biology_records[i].perturbation else None),
                            "relations": [asdict(r) for r in biology_records[i].relations
                                          if r.support_weight(config.biology_evidence_weight_policy) > 0]}
                           for i in np.r_[train, val]]
        if scaler.biology_vocabulary is None or scaler.biology_vocabulary["train_ids"] != dataset.ids[train].tolist():
            raise ValueError("Biological vocabulary does not match the frozen training allocation")
        if scaler.biology_vocabulary["evidence_weight_policy"] != config.biology_evidence_weight_policy:
            raise ValueError("Biological support-weight policy differs from the frozen configuration")
        if resume and payload.get("biology_input_binding") != biology_binding:
            raise ValueError("Resume biological input annotations differ")
    if config.use_library:
        bank_path = directory / "library_context.npz"
        bank = LibraryBank.load(bank_path) if resume else fit_library_context(normalized, train)
        if not resume:
            bank.save(bank_path)
        normalized = attach_library_context(normalized, bank)
    chemical_anchor_data = None
    if config.biology_kernel_mode != "off":
        chemical_anchor_data = fit_training_chemical_anchors(dataset, train, config)
        anchor_path = directory / "chemical_anchors.json"
        if resume:
            recorded_anchors = json.loads(anchor_path.read_text())
            if (chemical_anchor_data != recorded_anchors or
                    payload["model_config"].get("chemical_anchor_data") != recorded_anchors):
                raise ValueError("Resume chemical training structures or anchor binding differ")
        else:
            anchor_path.write_text(json.dumps(chemical_anchor_data, indent=2, allow_nan=False) + "\n")
    kwargs = model_kwargs(normalized, config, chemical_anchor_data=chemical_anchor_data)
    model = MeasurementWorldModel(**kwargs).to(config.device)
    model.set_outcome_transform(scaler.y_center, scaler.y_scale)
    encoder_policy = config.resolved_encoder_policy
    encoder_pretraining = None
    if encoder_policy == "frozen_random":
        from .model import GroupedProfileEncoder
        # Same initial encoder seed as the JEPA arms, frozen before learning.
        seed_everything(config.seed+101,config.threads)
        model.profile_encoder=GroupedProfileEncoder(kwargs["feature_groups"],config.hidden_dim,
            attention_layers=config.group_attention_layers,attention_heads=config.attention_heads)
    if encoder_policy.startswith("pretrained"):
        protected = set(dataset.ids[np.r_[splits["validation"], splits["calibration"], splits["evaluation"]]])
        encoder_pretraining = load_pretrained_encoder(model.profile_encoder,
            config.pretrained_encoder_checkpoint, feature_names=dataset.feature_names.tolist(),
            input_center=scaler.y_center, input_scale=scaler.y_scale, protected_ids=protected)
        if resume and payload.get("encoder_pretraining") != encoder_pretraining:
            raise ValueError("Resume pretrained encoder provenance differs")
    if encoder_policy in {"jepa_frozen", "jepa_finetune"}:
        if not resume:
            binding=_jepa_binding(normalized,train,scaler,config)
            if shared_jepa_directory is not None:
                _load_shared_jepa(model,shared_jepa_directory,directory,binding,kwargs)
            else:
                model.profile_encoder = fit_jepa(normalized, train, config, directory, kwargs)
                jepa_payload=torch.load(directory/"jepa.pt",map_location="cpu",weights_only=True)
                jepa_payload["binding"]=binding
                torch.save(jepa_payload,directory/"jepa.pt")
        if encoder_policy == "jepa_finetune":
            for parameter in model.profile_encoder.parameters():
                parameter.requires_grad_(True)
    if encoder_policy in {"jepa_frozen", "frozen_random", "pretrained_frozen"}:
        model.freeze_encoder()
    if not resume:
        if shared_jepa_directory is not None:
            paired=torch.load(Path(shared_jepa_directory)/"initial_state.pt",map_location="cpu",weights_only=True)
            current=model.state_dict()
            if paired.keys()!=current.keys() or any(not torch.equal(current[name].detach().cpu(),paired[name]) for name in current):
                raise ValueError("Paired objective arms have different initial model tensors")
        # The paired experiment compares these tensors directly, including all
        # fixed-coordinate buffers and the reused JEPA encoder.
        torch.save(model.state_dict(),directory/"initial_state.pt")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=config.learning_rate, weight_decay=config.weight_decay)
    steps_per_epoch = math.ceil(len(train) / config.batch_size)
    scheduler = make_world_model_scheduler(optimizer, config, steps_per_epoch)
    rng = np.random.default_rng(config.seed + 29)
    best, stale, start_epoch = float("inf"), 0, 0
    if resume:
        model.load_state_dict(payload["state_dict"])
        assert_artifact_geometry(model,scaler,payload)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        torch.set_rng_state(payload["torch_rng"])
        rng.bit_generator.state = payload["episode_rng"]
        start_epoch, best, stale = payload["epoch"], payload["best_validation"], payload["stale"]
    begin = time.monotonic()
    fitting_rows = np.r_[train, val]
    fitting_sources = set(dataset.groups[fitting_rows, :, 0][dataset.well_mask[fitting_rows]].tolist())
    fitting_sources_known = bool(fitting_sources) and -1 not in fitting_sources
    fitting_sources.discard(-1)
    if encoder_pretraining is not None:
        if "pretraining_source_groups" in encoder_pretraining:
            fitting_sources.update(encoder_pretraining["pretraining_source_groups"])
        else:
            fitting_sources_known = False
    log_event(progress, "world_model_start", train_compounds=len(train), validation_compounds=len(val),
              feature_dimension=dataset.Y.shape[-1], parameter_count=sum(p.numel() for p in model.parameters()),
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
              kernel=config.kernel_mode, encoder_policy=encoder_policy,
              zero_context_fraction=config.zero_context_fraction,
              objective=config.objective,utility_crps_weight=config.utility_crps_weight,
              utility_crps_samples=config.utility_crps_samples,paired_objective_rng=config.paired_objective_rng,
              biology_kernel=config.biology_kernel_mode, observation_family=config.observation_family,
              lr_schedule=config.lr_schedule, maximum_optimizer_steps=config.epochs*steps_per_epoch,
              checkpoint_selection="fixed_role_predictive_nll",
              initial_state_artifact="initial_state.pt")
    for epoch in range(start_epoch, config.epochs):
        if stale >= config.patience:
            break
        model.train()
        if encoder_policy in {"jepa_frozen", "frozen_random", "pretrained_frozen"}:
            model.profile_encoder.eval()
        total, predictive_total, base_total, count = 0.0, 0.0, 0.0, 0
        crps_total,crps_count=0.0,0
        crps_actions=np.zeros(3)
        gradient_norm_total, clipped_steps, optimizer_steps = 0., 0, 0
        monitor_modules = (epoch == 0 or (epoch + 1) % config.diagnostic_interval == 0)
        module_gradient_sum = {name: 0. for name in
                               ("profile_encoder", "chemical_prior", "chemistry_response_kernel",
                                "mean_head", "scale_head")}
        for minibatch,ix in enumerate(batches(rng.permutation(train), config.batch_size)):
            item = random_training_batch(normalized, ix, rng, config)
            optimizer.zero_grad(set_to_none=True)
            with objective_random_stream(config,epoch,minibatch):
                result = model.loss(item["inputs"], item["target_y"], item["target_mask"],objective=config.objective)
                loss = result["loss"]
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite measurement-model loss")
                loss.backward()
            base_value=float(loss.detach())
            predictive_total += float(result["nll"].detach()) * len(ix)
            # Release the large likelihood graph before building the utility
            # graph; gradients accumulate before the SAME optimizer step.
            del loss,result,item
            auxiliary_value=0.0
            if config.utility_crps_weight:
                inputs,y,mask=fixed_batch(normalized,ix,config)
                with objective_random_stream(config,epoch,minibatch,auxiliary=True):
                    result=derived_utility_crps(model,inputs,y,mask,samples=config.utility_crps_samples,
                                               chunk_size=min(config.mc_chunk_size,8))
                    auxiliary=config.utility_crps_weight*result["utility_crps"]
                    if not torch.isfinite(auxiliary):
                        raise FloatingPointError("Nonfinite derived utility CRPS")
                    auxiliary.backward()
                auxiliary_value=float(auxiliary.detach())
                scored=int(result["utility_scored_compounds"])
                crps_total+=float(result["utility_crps"].detach())*scored
                crps_actions+=result["utility_crps_by_action"].detach().cpu().numpy()*scored
                crps_count+=scored
                del auxiliary,result,inputs,y,mask
            if monitor_modules:
                for name in module_gradient_sum:
                    module = getattr(model, name, None)
                    if module is not None:
                        terms = [p.grad.detach().double().square().sum() for p in module.parameters()
                                 if p.grad is not None]
                        if terms:
                            module_gradient_sum[name] += float(torch.stack(terms).sum().sqrt())
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip,
                                                           error_if_nonfinite=True)
            gradient_norm_total += float(gradient_norm)
            clipped_steps += int(float(gradient_norm) > config.gradient_clip)
            optimizer_steps += 1
            optimizer.step()
            if config.lr_schedule == "cosine":
                scheduler.step()
            total += (base_value+auxiliary_value)*len(ix)
            base_total+=base_value*len(ix)
            count += len(ix)
        vloss = validation_loss(model, normalized, val, config)
        if config.lr_schedule == "plateau":
            scheduler.step(vloss)
        improved = vloss < best - config.min_delta
        best, stale = (vloss, 0) if improved else (best, stale + 1)
        payload = {"state_dict": model.state_dict(), "model_config": kwargs,
                   "train_config": asdict(config), "optimizer": optimizer.state_dict(),
                  "scheduler": scheduler.state_dict(), "epoch": epoch+1,
                   "optimizer_steps": (epoch+1)*steps_per_epoch,
                   "best_validation": best, "stale": stale, "torch_rng": torch.get_rng_state(),
                   "episode_rng": rng.bit_generator.state, "train_ids": dataset.ids[train].tolist(),
                   "validation_ids": dataset.ids[val].tolist(), "feature_names": dataset.feature_names.tolist(),
                   "encoder_pretraining": encoder_pretraining,
                   "initial_state_artifact":"initial_state.pt",
                   "checkpoint_selection":"fixed_role_predictive_nll",
                   "fitting_source_groups": sorted(fitting_sources),
                   "fitting_sources_known": fitting_sources_known}
        if config.use_biology_prior:
            payload["biology_input_binding"] = biology_binding
        torch.save(payload, directory / "last.pt")
        if improved:
            torch.save(payload, directory / "best.pt")
        log_event(progress, "world_model_epoch", epoch=epoch+1, train_objective=total/count,
                  train_nll=predictive_total/count,
                  train_base_objective=base_total/count,
                  train_utility_crps=(crps_total/crps_count if crps_count else None),
                  train_utility_crps_by_action=(crps_actions/crps_count).tolist() if crps_count else None,
                  train_utility_crps_scored_compounds=crps_count,
                  train_utility_crps_contribution=(total-base_total)/count,
                  gradient_norm_before_clip_mean=gradient_norm_total/optimizer_steps,
                  gradient_clipped_step_fraction=clipped_steps/optimizer_steps,
                  module_gradient_norms_before_clip=({k: v/optimizer_steps for k,v in module_gradient_sum.items()}
                                                     if monitor_modules else None),
                  validation_nll=vloss, best_validation_nll=best, stale=stale,
                  learning_rate=optimizer.param_groups[0]["lr"], elapsed_seconds=time.monotonic()-begin)
        if stale >= config.patience:
            break
    checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    bind_fitting_provenance(model, checkpoint["train_ids"], checkpoint["validation_ids"],
                           pretraining_ids=(encoder_pretraining or {}).get("pretraining_ids", ()),
                           fitting_source_groups=checkpoint.get("fitting_source_groups"),
                           fitting_sources_known=checkpoint.get("fitting_sources_known", False),
                           pretraining_sources_unknown=(encoder_pretraining is not None and
                                                        "pretraining_source_groups" not in encoder_pretraining))
    if config.use_library:
        model.library_bank = bank
    model.eval()
    return model, scaler


def load_model(directory, *, device="cpu"):
    if device != "cpu":
        raise ValueError("This exact-likelihood implementation requires CPU")
    directory = Path(directory)
    payload = torch.load(directory / "best.pt", map_location=device, weights_only=True)
    model = MeasurementWorldModel(**payload["model_config"]).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    config = TrainConfig(**payload["train_config"]).validate()
    if model.observation_family != config.observation_family or model.biology_kernel_mode != config.biology_kernel_mode:
        raise ValueError("Checkpoint model family or response kernel differs from its training configuration")
    if config.resolved_encoder_policy in {"jepa_frozen", "frozen_random", "pretrained_frozen"}:
        model.freeze_encoder()
    if "train_ids" not in payload or "validation_ids" not in payload:
        raise ValueError("Checkpoint lacks training/validation provenance; cannot certify a new split")
    model.eval()
    scaler = TrainScaler.load(directory / "scaler.json")
    if config.biology_kernel_mode != "off":
        anchor_data = json.loads((directory / "chemical_anchors.json").read_text())
        if anchor_data != payload["model_config"].get("chemical_anchor_data"):
            raise ValueError("Checkpoint and persisted chemical anchors differ")
        if anchor_data["train_ids"] != payload["train_ids"]:
            raise ValueError("Chemical anchors contain nontraining fitting identities")
    if config.use_biology_prior:
        if scaler.biology_vocabulary is None:
            raise ValueError("Enabled biological prior is missing its training vocabulary")
        if payload["model_config"].get("biology_vocabulary") != scaler.biology_vocabulary:
            raise ValueError("Checkpoint and scaler biological vocabularies differ")
        if scaler.biology_vocabulary["train_ids"] != payload["train_ids"]:
            raise ValueError("Biological vocabulary contains nontraining fitting identities")
        if scaler.biology_vocabulary["evidence_weight_policy"] != config.biology_evidence_weight_policy:
            raise ValueError("Checkpoint biological support-weight policy differs from configuration")
    elif payload["model_config"].get("use_biology_prior", False) or scaler.biology_vocabulary is not None:
        raise ValueError("Disabled biological prior has inconsistent fitted artifacts")
    pretraining = payload.get("encoder_pretraining")
    if config.resolved_encoder_policy.startswith("pretrained"):
        if not isinstance(pretraining, Mapping):
            raise ValueError("Pretrained checkpoint lacks retained encoder pretraining provenance")
        pretraining = validate_pretrained_encoder_payload(model.profile_encoder,
            dict(pretraining, encoder_state_dict=model.profile_encoder.state_dict()),
            feature_names=payload["feature_names"], input_center=scaler.y_center,
            input_scale=scaler.y_scale, protected_ids=payload["validation_ids"])
    elif pretraining is not None:
        raise ValueError("Nonpretrained checkpoint carries inconsistent pretraining provenance")
    source_known = payload.get("fitting_sources_known", False)
    source_groups = payload.get("fitting_source_groups")
    if pretraining is not None and "pretraining_source_groups" not in pretraining:
        source_known = False
    elif pretraining is not None and source_groups is not None:
        if not set(pretraining["pretraining_source_groups"]).issubset(source_groups):
            raise ValueError("Checkpoint fitting sources omit pretraining source groups")
    bind_fitting_provenance(model, payload["train_ids"], payload["validation_ids"],
                           pretraining_ids=(pretraining or {}).get("pretraining_ids", ()),
                           fitting_source_groups=source_groups,
                           fitting_sources_known=source_known,
                           pretraining_sources_unknown=(pretraining is not None and
                                                        "pretraining_source_groups" not in pretraining))
    if tuple(scaler.train_ids) != model.training_ids:
        raise ValueError("Checkpoint and fitted scaler training identifiers differ")
    assert_artifact_geometry(model,scaler,payload)
    if config.use_library:
        bank_path = directory / "library_context.npz"
        if not bank_path.exists():
            raise ValueError("Checkpoint declares library context but its fitted bank is missing")
        model.library_bank = LibraryBank.load(bank_path)
        if not set(model.library_bank.ids).issubset(model.training_ids):
            raise ValueError("Library bank contains non-training outcome identities")
    return model, scaler, config, payload
