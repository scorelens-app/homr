import os
from time import perf_counter

import numpy as np
from PIL import Image

from homr.export import EXPORT
from homr.simple_logging import eprint
from homr.transformer.configs import Config
from homr.transformer.decoder_inference import get_decoder
from homr.transformer.encoder_inference import Encoder
from homr.transformer.vocabulary import EncodedSymbol, _split_articulations
from homr.type_definitions import NDArray


class Staff2Score:
    """
    Inference class for Tromr. Use predict() for prediction
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.encoder = Encoder(self.config)
        self.decoder = get_decoder(self.config)

        if not os.path.exists(self.config.filepaths.rhythmtokenizer):
            raise RuntimeError(
                "Failed to find tokenizer config" + self.config.filepaths.rhythmtokenizer
            )  # noqa: E501

    def predict(self, image: NDArray) -> list[EncodedSymbol]:
        """
        Inference an image (NDArray) using Tromr.
        """
        x = _transform(image=image)

        t0 = perf_counter()

        # Create special tokens
        start_token = np.array([[1]], dtype=np.int64)
        nonote_token = np.array([[0]], dtype=np.int64)

        # Generate context with encoder. The encoder and decoder may run in
        # different precisions (e.g. the CoreML encoder is fp16 while the
        # decoder stays on the fp32 CPU model), so cast the context to the
        # dtype the decoder expects before handing it over.
        context = self.encoder.generate(x)
        context_dtype = np.float16 if self.decoder.fp16 else np.float32
        if context.dtype != context_dtype:
            context = context.astype(context_dtype)

        # Make a prediction using decoder
        out = self.decoder.generate(
            start_token,
            nonote_token,
            seq_len=self.config.max_seq_len,
            eos_token=self.config.eos_token,
            context=context,
        )

        eprint(f"Inference Time Tromr: {perf_counter()-t0}")

        return fold_to_455_tokens(out)


def _fold_articulation(articulation: str) -> str:
    parts = [p for p in _split_articulations(articulation) if not p.startswith("dynamic_")]
    parts = sorted({"accent" if p == "accentStrong" else p for p in parts})
    return "_".join(parts) if parts else ("_" if articulation not in (".", "") else articulation)


def fold_to_455_tokens(symbols: list[EncodedSymbol]) -> list[EncodedSymbol]:
    """Probe of model run 509/512 on the 455-era pipeline: map the newer vocabulary onto the tokens the
    pipeline knows. Voice-2 positions fold into their staff (upper2 -> upper, lower2 -> lower), dynamics
    tokens are dropped (marked removed in the export), accentStrong reads as accent and dynamic modifiers
    leave the articulation."""
    result: list[EncodedSymbol] = []
    dropped: list[str] = []
    for symbol in symbols:
        if symbol.rhythm.startswith("dynamic_"):
            if symbol.export_id:
                dropped.append(symbol.export_id)
            continue
        if symbol.position in ("upper2", "lower2"):
            symbol.position = symbol.position[:-1]
        if "dynamic_" in symbol.articulation or "accentStrong" in symbol.articulation:
            symbol.articulation = _fold_articulation(symbol.articulation)
        result.append(symbol)
    if dropped and EXPORT.on():
        EXPORT.mark(dropped, "removed", reason="dynamic_token")
    return result


class ConvertToArray:
    def __init__(self) -> None:
        self.mean = np.array([0.7931]).reshape(1, 1, 1)
        self.std = np.array([0.1738]).reshape(1, 1, 1)

    def normalize(self, array: NDArray) -> NDArray:
        return (array - self.mean) / self.std

    def __call__(self, image: NDArray) -> NDArray:
        arr = np.array(image) / 255
        arr = arr[np.newaxis, np.newaxis, :, :]
        return self.normalize(arr).astype(np.float32)


_transform = ConvertToArray()


def test_transformer_on_image(path_to_img: str) -> None:
    """
    Tests the transformer on an image and prints the results.
    Args:
        path_to_img(str): Path to the image to test
    """

    model = Staff2Score(Config())
    image = Image.open(path_to_img)
    out = model.predict(np.array(image))
    eprint(out)


if __name__ == "__main__":
    import sys

    test_transformer_on_image(sys.argv[1])
