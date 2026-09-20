"""Benchmark a model/detector on a labelled face dataset and calibrate the threshold.

    python -m videokyc.face.evaluate --data path/to/lfw --model ArcFace --pairs 1000 --degrade

Dataset layout (LFW / any identity-folder dataset):  DATA/<person>/<image>.jpg
``--degrade`` down-scales and JPEG-compresses the *reference* image of each pair
to mimic the tiny, compressed portrait on an Aadhaar card.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import cv2
import numpy as np

from . import detection
from .config import FaceConfig
from .embedding import DeepFaceEmbedder
from .matching import cosine_distance
from .metrics import roc_curve, summarize

IMG_EXT = {".jpg", ".jpeg", ".png"}


def degrade(img: np.ndarray, width: int = 160, jpeg_quality: int = 45) -> np.ndarray:
    """Simulate an Aadhaar-style portrait: small, then lossy-compressed."""
    scale = width / img.shape[1]
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def build_pairs(index: dict[str, list[Path]], n_pairs: int, seed: int) -> tuple[list, list]:
    rng = random.Random(seed)  # noqa: S311 - seeded for reproducible pairs
    multi = [p for p, imgs in index.items() if len(imgs) >= 2]
    people = list(index)
    if not multi or len(people) < 2:
        raise SystemExit("Dataset needs >=2 identities and >=1 identity with >=2 images.")

    genuine: list[tuple[Path, Path]] = []
    for person in itertools.islice(itertools.cycle(rng.sample(multi, len(multi))), n_pairs):
        genuine.append(tuple(rng.sample(index[person], 2)))
    impostor: list[tuple[Path, Path]] = []
    while len(impostor) < n_pairs:
        a, b = rng.sample(people, 2)
        impostor.append((rng.choice(index[a]), rng.choice(index[b])))
    return genuine, impostor


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--model", default="ArcFace")
    ap.add_argument("--detector", default="yunet", choices=["yunet", "haar"])
    ap.add_argument("--alignment", default="auto", choices=["auto", "arcface", "box"])
    ap.add_argument("--pairs", type=int, default=500, help="genuine pairs (same number of impostor pairs)")
    ap.add_argument("--degrade", action="store_true", help="simulate Aadhaar-quality reference photos")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("results"))
    args = ap.parse_args(argv)

    cfg = FaceConfig(model_name=args.model, detector=args.detector, alignment=args.alignment)
    detector, embedder = detection.build_detector(cfg), DeepFaceEmbedder(args.model)

    index = {
        d.name: sorted(f for f in d.iterdir() if f.suffix.lower() in IMG_EXT)
        for d in sorted(args.data.iterdir()) if d.is_dir()
    }
    genuine_pairs, impostor_pairs = build_pairs(index, args.pairs, args.seed)

    cache: dict[tuple[Path, bool], np.ndarray | None] = {}
    skipped = 0

    def embed(path: Path, is_reference: bool) -> np.ndarray | None:
        key = (path, is_reference)
        if key not in cache:
            img = cv2.imread(str(path))
            if img is not None and is_reference and args.degrade:
                img = degrade(img)
            box = detection.best_face(detector.detect(img)) if img is not None else None
            cache[key] = None if box is None else embedder.embed(
                detection.align_face(img, box, cfg)
            )
        return cache[key]

    def distances(pairs: list) -> np.ndarray:
        nonlocal skipped
        out = []
        for a, b in pairs:
            ea, eb = embed(a, True), embed(b, False)
            if ea is None or eb is None:
                skipped += 1
                continue
            out.append(cosine_distance(ea, eb))
        return np.array(out)

    genuine, impostor = distances(genuine_pairs), distances(impostor_pairs)
    report = {
        "model": args.model, "detector": args.detector, "alignment": args.alignment,
        "degraded_reference": args.degrade,
        "pairs_skipped_no_face": skipped, "default_threshold": cfg.effective_threshold,
        **summarize(genuine, impostor),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.model}_{args.detector}_{args.alignment}{'_degraded' if args.degrade else ''}"
    (args.out / f"{stem}.json").write_text(json.dumps(report, indent=2))
    _plot(args.out / f"{stem}_roc.png", genuine, impostor, args.model)
    print(json.dumps(report, indent=2))


def _plot(path: Path, genuine: np.ndarray, impostor: np.ndarray, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    _, far, tar = roc_curve(genuine, impostor)
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    a.plot(far, tar)
    a.plot([0, 1], [0, 1], "k--", lw=0.5)
    a.set(xlabel="False accept rate", ylabel="True accept rate", title=f"ROC - {title}")
    b.hist(genuine, bins=40, alpha=0.6, label="genuine")
    b.hist(impostor, bins=40, alpha=0.6, label="impostor")
    b.set(xlabel="Cosine distance", title="Distance distributions")
    b.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)


if __name__ == "__main__":
    main()
