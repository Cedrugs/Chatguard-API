from typing import Dict, Optional, List
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from .config import settings


class ModelBundle:
    def __init__(self) -> None:
        self.pipe: Optional[TextClassificationPipeline] = None
        self.id2label: Dict[int, str] = {}
        self.device_str: str = "cpu"

    def _resolve_device(self) -> str:
        if settings.device in {"cpu", "cuda"}:
            if settings.device == "cuda" and not torch.cuda.is_available():
                return "cpu"
            return settings.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        self.device_str = self._resolve_device()
        device_index = 0 if self.device_str == "cuda" else -1

        tok = AutoTokenizer.from_pretrained(settings.model_id)
        mdl = AutoModelForSequenceClassification.from_pretrained(
            settings.model_id)

        self.id2label = {int(k): v for k, v in getattr(mdl.config, "id2label", {}).items()} or {
            i: f"LABEL_{i}" for i in range(mdl.config.num_labels)
        }

        self.pipe = TextClassificationPipeline(
            model=mdl,
            tokenizer=tok,
            device=device_index,
            top_k=None,
            truncation=True,
            padding=True,
            max_length=settings.max_length,
            function_to_apply=None
        )
        _ = self.pipe(["warmup"])  # reduce first-hit latency

    @staticmethod
    def _softmax(logits: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(logits, dim=-1)

    def _probs_dict(self, row: List[float]) -> Dict[str, float]:
        out = {}
        for idx, p in enumerate(row):
            name = self.id2label.get(idx, str(idx)).lower()
            out[name] = float(p)
        
        if "toxic" in out and "clean" not in out:
            out["clean"] = 1.0 - out["toxic"]
        elif "clean" in out and "toxic" not in out:
            out["toxic"] = 1.0 - out["clean"]
        elif "toxic" not in out and "clean" not in out:
            if len(out) == 2:
                labels = list(out.keys())
                out["toxic"] = out[labels[1]]
                out["clean"] = out[labels[0]]
            else:
                out["toxic"] = 0.0
                out["clean"] = 1.0
        
        return {"clean": out["clean"], "toxic": out["toxic"]}

    def predict_one(self, text: str, threshold: Optional[float]) -> Dict:
        assert self.pipe is not None
        tok = self.pipe.tokenizer
        mdl = self.pipe.model
        inputs = tok(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_length,
        )
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
        with torch.no_grad():
            logits = mdl(**inputs).logits
        probs = self._softmax(logits)[0].tolist()
        probs_dict = self._probs_dict(probs)
        label = "toxic" if (probs_dict["toxic"] >= probs_dict["clean"]) else "clean"
        if threshold is not None:
            label = "toxic" if probs_dict["toxic"] >= threshold else "clean"
        return {"label": label, "probs": probs_dict}

    def predict_batch(self, texts: List[str], threshold: Optional[float]) -> List[Dict]:
        assert self.pipe is not None
        tok = self.pipe.tokenizer
        mdl = self.pipe.model
        inputs = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=settings.max_length,
        )
        inputs = {k: v.to(self.device_str) for k, v in inputs.items()}
        with torch.no_grad():
            logits = mdl(**inputs).logits
        probs = self._softmax(logits).tolist()
        results = []
        for row in probs:
            probs_dict = self._probs_dict(row)
            label = "toxic" if (
                probs_dict["toxic"] >= probs_dict["clean"]) else "clean"
            if threshold is not None:
                label = "toxic" if probs_dict["toxic"] >= threshold else "clean"
            results.append({"label": label, "probs": probs_dict})
        return results


model_bundle = ModelBundle()
