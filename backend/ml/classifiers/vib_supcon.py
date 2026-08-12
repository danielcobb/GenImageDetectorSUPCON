import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

from ml.classifiers.pytorch_base import PyTorchClassifier
from ml.models.VIBSupCon.clip import clip as openai_clip

# CLIP ViT-L/14 normalization (same constants as ml.classifiers.vib, but
# kept local since these two VIB variants load CLIP through unrelated code
# paths - see VIBSupConEncoder docstring).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CLIP_IMAGE_EMBED_DIM = 768  # ViT-L/14 encode_image() output dim
VIB_SUPCON_K = 256


class VIBSupConEncoder(nn.Module):
    """CLIP ViT-L/14 + variational information bottleneck encoder, vendored
    from the VIBAIGCDetect SupCon training pipeline (main_linear_VIB.py /
    VIBEncoder).

    Unlike ml.classifiers.vib.VIBNet - whose backbone stays frozen at stock
    HuggingFace CLIP weights - this backbone was fully fine-tuned during
    SupCon pretraining, and the checkpoint stores it under OpenAI's own
    CLIP parameter names/layout (from openai/CLIP's clip.py + model.py,
    vendored under ml/models/VIBSupCon/clip/), not HuggingFace
    transformers'. The two implementations are not just differently named
    but structurally different (e.g. fused vs. split QKV attention
    projections), so reusing VIBNet's HF-based backbone here would silently
    fail to load most of the fine-tuned weights. Loading CLIP the "wrong"
    way for this specific checkpoint is a deliberate, checkpoint-driven
    departure from the HF-transformers convention used elsewhere in this
    codebase (Effort, VIB) - not an oversight.
    """

    def __init__(
        self,
        clip_name: str = "ViT-L/14",
        k: int = VIB_SUPCON_K,
        download_root: str | None = None,
    ):
        super().__init__()
        if clip_name != "ViT-L/14":
            raise ValueError(
                f"VIBSupConEncoder only supports ViT-L/14 (got {clip_name!r})"
            )
        self.k = k
        self.model, _ = openai_clip.load(
            clip_name, device="cpu", download_root=download_root
        )
        self.fc_1 = nn.Linear(CLIP_IMAGE_EMBED_DIM, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.fc_2 = nn.Linear(1024, 1024)
        self.fc_3 = nn.Linear(1024, 2 * k)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.encode_image(x)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        features = features.float()
        features = self.dropout(features)
        stats = self.fc_1(features)
        stats = self.relu(stats)
        stats = self.fc_2(stats)
        stats = self.relu(stats)
        stats = self.fc_3(stats)
        mu = stats[:, : self.k]
        std = F.softplus(stats[:, self.k :] - 5, beta=1)
        return mu, std


class VIBSupConLinearProbe(nn.Module):
    """Phase 2 linear probe on top of the (post-SupCon) VIB encoder.

    Decodes the posterior mean `mu` rather than a reparameterized sample -
    the standard VIB test-time estimate, and the same deterministic-
    inference choice ml.classifiers.vib.VIBNet already makes, so a given
    image scores the same every time instead of picking up sampling noise.
    """

    def __init__(self, encoder: VIBSupConEncoder, k: int = VIB_SUPCON_K):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Linear(k, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encoder(x)
        return self.fc(mu).squeeze(1)  # logit, shape [batch]


class VIBSupConClassifier(PyTorchClassifier):
    """VIB-Net + SupCon variant - CLIP ViT-L/14 (fully fine-tuned) + VIB
    head, pretrained with supervised contrastive learning on a 4-generator
    combined dataset (adm, biggan, midjourney, sd_v1_4), then linear-probed.

    Takes a single checkpoint, "vib_linear_5.pth", whose top-level 'model'
    key holds the FULL VIBSupConLinearProbe state dict (encoder + linear
    head; 454 tensors), not just the head. Confirmed against the real
    uploaded checkpoint - loading it produces zero missing and zero
    unexpected keys against this architecture. There's no fallback to
    "vib_supcon_3.pth" (the raw Phase 1 SupCon-pretrained encoder): that
    file is a training artifact, never read at inference time. (An earlier
    version of this class accepted both paths and had ~60 lines of
    untested fallback logic for the case where the linear checkpoint might
    only contain the head - that branch turned out to be unreachable once
    real weights landed, and eagerly downloading the 4.17GB encoder
    checkpoint for a file that's never opened isn't a good trade on the
    8GB EC2 instance this runs on. If a future checkpoint export changes
    shape, the error in load_weights will say so explicitly rather than
    silently guessing.)

    Label convention / confidence direction (read this before touching
    postprocess): training used GenImageFolder, whose alphabetical
    class_to_idx is {'ai': 0, 'nature': 1} - label 1 = real. The linear
    head is trained against that label, so sigmoid(logit) is already
    P(real). This codebase's convention (see PyTorchClassifier.postprocess
    and every other classifier's postprocess/analyze) is that classifiers
    return confidence-that-the-image-is-real on a 0-100 scale, which is
    exactly what sigmoid(logit) * 100 already gives us - NO inversion to
    P(fake) is needed. (An earlier draft of this model's integration notes
    suggested inverting to report P(fake), reasoning that "the app reports
    confidence as probability of fake" - that premise doesn't match this
    codebase: routes.py's aggregate confidence and every sibling classifier
    treat the returned float as P(real). Inverting here would have
    silently flipped every prediction from this model. If you're re-deriving
    this, check analysis/routes.py's aggregate confidence calculation and
    ml/classifiers/vib.py's postprocess as ground truth, not this comment.)
    """

    def __init__(
        self,
        model_path: str,
        clip_download_root: str | None = None,
        device: str | None = None,
        quiet: bool = False,
    ):
        self.clip_download_root = clip_download_root or os.getenv(
            "VIB_SUPCON_CLIP_CACHE_DIR"
        )
        self.quiet = quiet

        super().__init__(model_path, device)

    def get_model_architecture(self) -> nn.Module:
        encoder = VIBSupConEncoder(
            clip_name="ViT-L/14",
            k=VIB_SUPCON_K,
            download_root=self.clip_download_root,
        )
        return VIBSupConLinearProbe(encoder, k=VIB_SUPCON_K)

    def get_transforms(self):
        # Must match VIBAIGCDetect's main_linear_VIB.py training transform
        # exactly: standard torchvision Resize(256)/CenterCrop(224), NOT
        # OpenAI CLIP's native Resize(224, BICUBIC) that
        # ml.classifiers.vib.VIBClassifier uses for the other VIB
        # checkpoint (a different training pipeline).
        return transforms.Compose(
            [
                transforms.Lambda(lambda image: image.convert("RGB")),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
            ]
        )

    def _prune_text_tower(self) -> None:
        """Drop the CLIP text tower after weights are loaded.

        encode_image() only touches self.visual (see openai/CLIP's
        model.py), and the text transformer is a meaningful chunk of a
        ViT-L/14's parameters we'd otherwise carry uselessly in RAM -
        worth doing given this model roughly doubles the CLIP backbone
        memory footprint already running in this service (see integration
        notes' memory warning). Must run AFTER load_state_dict: the
        checkpoint's state dict includes text-tower keys, and a strict-ish
        load needs them present to match against.
        """
        clip_model = self.model.encoder.model
        for attr in (
            "transformer",
            "token_embedding",
            "ln_final",
            "text_projection",
            "logit_scale",
        ):
            if hasattr(clip_model, attr):
                delattr(clip_model, attr)

    def load_weights(self):
        """Load the Phase 2 checkpoint's ckpt['model'] - the full
        encoder+fc state dict (454 tensors). Verified against the actual
        uploaded vib_linear_5.pth: zero missing/unexpected keys."""
        ckpt = torch.load(self.model_path, map_location="cpu", weights_only=False)

        if not (isinstance(ckpt, dict) and isinstance(ckpt.get("model"), dict)):
            raise RuntimeError(
                f"[VIB-SupCon] Expected a top-level 'model' key in {self.model_path} "
                "holding the full encoder+fc state dict (per main_linear_VIB.py's "
                "save_checkpoint()). Found top-level keys: "
                f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}."
            )

        state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
        result = self.model.load_state_dict(state, strict=False)

        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"[VIB-SupCon] Checkpoint {self.model_path} doesn't match the "
                f"model architecture - missing={result.missing_keys[:20]}, "
                f"unexpected={result.unexpected_keys[:20]}. This loading code was "
                "verified against the checkpoint at integration time; if this "
                "fires, the uploaded file has changed shape and needs "
                "re-verifying, not a silent partial load."
            )

        if not self.quiet:
            print(f"[VIB-SupCon] Loaded {len(state)} tensors from {self.model_path}.")

        self._prune_text_tower()

    def postprocess(self, output: torch.Tensor) -> float:
        # sigmoid(logit) is already P(real) - see class docstring.
        logit = output.view(-1)[0].item()
        prob_real = torch.sigmoid(torch.tensor(logit)).item()
        confidence = round(prob_real * 100, 1)
        if not self.quiet:
            print(f"[VIB-SupCon Debug] Logit: {logit:.4f}, Prob(real): {prob_real:.4f}")
        return confidence
