"""Explicit model and training choices; no hidden experimental searches."""
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


@dataclass
class TrainConfig:
    seed: int = 20260911
    epochs: int = 100
    jepa_epochs: int = 60
    batch_size: int = 32
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    patience: int = 15
    min_delta: float = 0.0001
    gradient_clip: float = 5.0
    hidden_dim: int = 256
    latent_rank: int = 32
    residual_rank: int = 8
    group_attention_layers: int = 2
    group_assignment: str = "cellprofiler"
    attention_heads: int = 4
    reference_loss_weight: float = 0.1
    latent_kl_weight: float = 1.0
    chemical_regularization_weight: float = 0.0
    use_chemistry: bool = True
    use_references: bool = True
    use_library: bool = True
    use_biology_prior: bool = False
    biology_evidence_weight_policy: str = "confidence_or_unit_support"
    kernel_mode: str = "measurement"
    use_jepa: bool = True
    encoder_policy: str = "auto"
    pretrained_encoder_checkpoint: str | None = None
    zero_context_fraction: float = 0.1
    device: str = "cpu"
    threads: int = 4
    samples: int = 2000
    mc_chunk_size: int = 32
    reference_access: str = "observed_only"
    objective: str = "elbo"
    utility_crps_weight: float = 0.0
    utility_crps_samples: int = 16
    paired_objective_rng: bool = False
    biology_kernel_mode: str = "off"
    biology_kernel_anchors: int = 64
    observation_family: str = "gaussian"
    lr_schedule: str = "plateau"
    warmup_steps: int = 60
    min_learning_rate: float = 0.000003
    diagnostic_interval: int = 5

    def validate(self):
        if self.biology_kernel_mode not in {"off", "structured", "generic", "mlp"}:
            raise ValueError("Unknown chemical response kernel mode")
        if (isinstance(self.biology_kernel_anchors, bool) or
                not isinstance(self.biology_kernel_anchors, int) or self.biology_kernel_anchors < 1):
            raise ValueError("biology_kernel_anchors must be a positive integer")
        if self.biology_kernel_mode != "off" and not self.use_chemistry:
            raise ValueError("A chemical response kernel requires chemical inputs")
        if self.observation_family not in {"gaussian", "copula_t4"}:
            raise ValueError("Unknown joint observation family")
        if self.observation_family != "gaussian" and self.objective != "predictive_nll":
            raise ValueError("The copula observation model requires its full predictive_nll objective")
        if self.lr_schedule not in {"plateau", "cosine"}:
            raise ValueError("Unknown learning-rate schedule")
        if (isinstance(self.warmup_steps, bool) or not isinstance(self.warmup_steps, int)
                or self.warmup_steps < 0):
            raise ValueError("warmup_steps must be a nonnegative integer")
        if not math.isfinite(self.min_learning_rate) or self.min_learning_rate <= 0:
            raise ValueError("min_learning_rate must be finite and positive")
        if self.lr_schedule == "cosine" and self.min_learning_rate > self.learning_rate:
            raise ValueError("The cosine minimum cannot exceed the peak learning rate")
        if not isinstance(self.use_biology_prior, bool):
            raise ValueError("use_biology_prior must be boolean")
        if self.biology_evidence_weight_policy not in {"confidence_or_unit_support", "supplied_confidence_only"}:
            raise ValueError("Declare the biological evidence-weight policy")
        for name in ("epochs", "jepa_epochs", "batch_size", "patience", "hidden_dim",
                     "latent_rank", "residual_rank", "threads", "samples",
                     "mc_chunk_size", "attention_heads", "diagnostic_interval"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.group_attention_layers,bool) or not isinstance(self.group_attention_layers,int) or self.group_attention_layers<0:
            raise ValueError("group_attention_layers must be a nonnegative integer")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if self.group_assignment not in {"cellprofiler", "shuffled"}:
            raise ValueError("Declare cellprofiler or shuffled feature grouping")
        if self.objective not in {"elbo", "predictive_nll"}:
            raise ValueError("Declare elbo or predictive_nll training")
        if isinstance(self.utility_crps_samples, bool) or not isinstance(self.utility_crps_samples, int) or self.utility_crps_samples < 2:
            raise ValueError("Fair CRPS needs at least two independent joint draws")
        if not isinstance(self.paired_objective_rng, bool):
            raise ValueError("paired_objective_rng must be boolean")
        if not math.isfinite(self.utility_crps_weight) or self.utility_crps_weight < 0:
            raise ValueError("utility_crps_weight must be finite and nonnegative")
        if self.utility_crps_weight and self.objective != "predictive_nll":
            raise ValueError("The declared utility-score arm augments predictive_nll")
        if self.encoder_policy not in {"auto", "trainable", "frozen_random", "jepa_frozen", "jepa_finetune", "pretrained_frozen", "pretrained_finetune"}:
            raise ValueError("Declare the encoder training policy")
        if self.resolved_encoder_policy.startswith("pretrained") and not self.pretrained_encoder_checkpoint:
            raise ValueError("A real pretrained checkpoint must be supplied")
        if not 0 <= self.zero_context_fraction < 1:
            raise ValueError("zero_context_fraction must be in [0,1)")
        for name in ("reference_loss_weight", "latent_kl_weight", "chemical_regularization_weight"):
            if not 0 <= getattr(self, name) < float("inf"):
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.latent_kl_weight != 1:
            raise ValueError("The conditional ELBO uses latent_kl_weight=1")
        if any(not math.isfinite(getattr(self,name)) for name in
               ("learning_rate","weight_decay","gradient_clip","min_delta")):
            raise ValueError("Optimizer and stopping values must be finite")
        if self.kernel_mode not in {"measurement", "generic", "mlp"}:
            raise ValueError("Unknown kernel mode")
        if self.reference_access not in {"observed_only", "all_declared", "none"}:
            raise ValueError("Reference access must be explicit")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.gradient_clip <= 0 or self.min_delta < 0:
            raise ValueError("Invalid optimizer configuration")
        return self

    @property
    def resolved_encoder_policy(self):
        if self.encoder_policy == "auto":
            return "jepa_frozen" if self.use_jepa else "trainable"
        return self.encoder_policy

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()
