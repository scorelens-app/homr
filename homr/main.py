import argparse
import glob
import os
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import Future
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
import onnxruntime as ort

from homr import color_adjust, download_utils
from homr.autocrop import autocrop
from homr.bar_line_detection import (
    detect_bar_lines,
    prepare_bar_line_image,
)
from homr.bounding_boxes import (
    BoundingEllipse,
    RotatedBoundingBox,
    create_bounding_ellipses,
    create_rotated_bounding_boxes,
)
from homr.brace_dot_detection import (
    find_braces_brackets_and_grand_staff_lines,
    prepare_brace_dot_image,
)
from homr import export
from homr.debug import Debug
from homr.export import EXPORT
from homr.model import InputPredictions, MultiStaff, Staff
from homr.music_xml_generator import XmlGeneratorArguments, generate_xml
from homr.noise_filtering import filter_predictions
from homr.note_detection import add_notes_to_staffs, combine_noteheads_with_stems
from homr.onnx_providers import coreml_available, cuda_available, rocm_available
from homr.pdf_utils import render_pdf_to_image
from homr.relieur import process_concat
from homr.resize import resize_image
from homr.segmentation.config import segnet_path_onnx, segnet_path_onnx_fp16
from homr.segmentation.inference_segnet import extract
from homr.simple_logging import eprint
from homr.staff_detection import break_wide_fragments, detect_staff, make_lines_stronger
from homr.staff_parsing import parse_staffs
from homr.staff_position_save_load import load_staff_positions, save_staff_positions
from homr.title_detection import detect_title, download_ocr_weights
from homr.transformer.configs import Config, default_config, root_dir
from homr.type_definitions import NDArray

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class PredictedSymbols:
    def __init__(
        self,
        noteheads: list[BoundingEllipse],
        staff_fragments: list[RotatedBoundingBox],
        clefs_keys: list[RotatedBoundingBox],
        stems_rest: list[RotatedBoundingBox],
        bar_lines: list[RotatedBoundingBox],
    ) -> None:
        self.noteheads = noteheads
        self.staff_fragments = staff_fragments
        self.clefs_keys = clefs_keys
        self.stems_rest = stems_rest
        self.bar_lines = bar_lines


class InvalidProgramArgumentException(Exception):
    """Raise this exception for issues which the user can address."""


class GpuSupport(Enum):
    No = "no"
    AUTO = "auto"
    FORCE = "force"


def get_predictions(
    original: NDArray,
    preprocessed: NDArray,
    img_path: str,
    enable_cache: bool,
    segnet_use_gpu: bool,
) -> InputPredictions:
    result = extract(
        preprocessed,
        img_path,
        step_size=320,
        use_cache=enable_cache,
        use_gpu_inference=segnet_use_gpu,
    )
    original_image = cv2.resize(original, (result.staff.shape[1], result.staff.shape[0]))
    preprocessed_image = cv2.resize(preprocessed, (result.staff.shape[1], result.staff.shape[0]))
    return InputPredictions(
        original=original_image,
        preprocessed=preprocessed_image,
        notehead=result.notehead.astype(np.uint8),
        symbols=result.symbols.astype(np.uint8),
        staff=result.staff.astype(np.uint8),
        clefs_keys=result.clefs_keys.astype(np.uint8),
        stems_rest=result.stems_rests.astype(np.uint8),
    )


def replace_extension(path: str, new_extension: str) -> str:
    return os.path.splitext(path)[0] + new_extension


def load_and_preprocess_predictions(
    image_path: str, enable_debug: bool, enable_cache: bool, segnet_use_gpu: bool
) -> tuple[InputPredictions, Debug]:
    image = cv2.imread(image_path)
    if image is None:
        raise InvalidProgramArgumentException(
            "The file format is not supported, please provide a JPG or PNG image file:" + image_path
        )
    input_image = image
    image = autocrop(image)
    crop = _crop_box(input_image, image)
    cropped_size = (image.shape[1], image.shape[0])
    image = resize_image(image)
    EXPORT.frame("input", {"width": input_image.shape[1], "height": input_image.shape[0]})
    EXPORT.stage("autocrop", crop=crop)
    preprocessed = color_adjust.apply_clahe(image)
    predictions = get_predictions(image, preprocessed, image_path, enable_cache, segnet_use_gpu)
    if EXPORT.on():
        seg_h, seg_w = predictions.staff.shape[:2]
        EXPORT.frame(
            "work",
            {
                "width": seg_w,
                "height": seg_h,
                "from_input": {
                    "crop": crop,
                    "scale": [seg_w / cropped_size[0], seg_h / cropped_size[1]],
                },
            },
        )
        EXPORT.write_maps("staffdetect", "staffdetect/seg_raw.npz", _seg_maps(predictions))
        EXPORT.stage("segnet")
    debug = Debug(predictions.original, image_path, enable_debug)
    debug.write_image("color_adjust", preprocessed)

    predictions = filter_predictions(predictions, debug)

    predictions.staff = make_lines_stronger(predictions.staff, (1, 2))
    if EXPORT.on():
        EXPORT.write_maps("staffdetect", "staffdetect/seg_filtered.npz", _seg_maps(predictions))
        EXPORT.stage("noise_filter")
    debug.write_threshold_image("staff", predictions.staff)
    debug.write_threshold_image("symbols", predictions.symbols)
    debug.write_threshold_image("stems_rest", predictions.stems_rest)
    debug.write_threshold_image("notehead", predictions.notehead)
    debug.write_threshold_image("clefs_keys", predictions.clefs_keys)
    return predictions, debug


def _crop_box(original: NDArray, cropped: NDArray) -> list[int]:
    """Where autocrop's result (a view into the original) sits in the original image."""
    if cropped.shape == original.shape:
        return [0, 0, original.shape[1], original.shape[0]]
    offset = cropped.__array_interface__["data"][0] - original.__array_interface__["data"][0]
    y, rest = divmod(offset, original.strides[0])
    x = rest // original.strides[1]
    return [int(x), int(y), int(x + cropped.shape[1]), int(y + cropped.shape[0])]


def _seg_maps(predictions: InputPredictions) -> dict[str, NDArray]:
    return {
        "staff": predictions.staff,
        "symbols": predictions.symbols,
        "stems_rest": predictions.stems_rest,
        "notehead": predictions.notehead,
        "clefs_keys": predictions.clefs_keys,
    }


def _export_candidates(**values: object) -> None:
    if not EXPORT.on("staffdetect"):
        return
    out: dict[str, object] = {}
    for key, value in values.items():
        if isinstance(value, list):
            out[key] = [export.rotated(getattr(v, "notehead", v)) for v in value]
        else:
            out[key] = value
    EXPORT.write_json("staffdetect", "staffdetect/candidates.json", out)


def _export_staffs(staffs: list[Staff]) -> list[dict[str, object]]:
    return [
        {
            "id": EXPORT.staff_id(s),
            "box": export.staff_box(s),
            "lines": export.staff_lines(s),
            "line_spacing": round(float(s.average_unit_size), 3),
            "grand": bool(s.is_grandstaff),
        }
        for s in staffs
    ]


def _export_geometry(
    multi_staffs: list[MultiStaff],
    notes: list[object],
    bar_line_boxes: list[RotatedBoundingBox] | None = None,
) -> None:
    if not EXPORT.on("geometry"):
        return
    staffs = []
    bar_line_staffs: list[list[str]] = [[] for _ in bar_line_boxes or []]
    bar_lines = [
        {
            "id": f"b{k}",
            "box": export.rotated(b),
            "x": round(float(b.center[0]), 2),
            "staffs": bar_line_staffs[k],
        }
        for k, b in enumerate(bar_line_boxes or [])
    ]
    for system, multi in enumerate(multi_staffs):
        for k, staff in enumerate(multi.staffs):
            entry = _export_staffs([staff])[0]
            entry["system"] = system
            entry["index_in_system"] = k
            symbols = []
            bar_line_ids = []
            # homr's own staff-zone test (the one it applies to noteheads)
            for k_bar, bar_box in enumerate(bar_line_boxes or []):
                if staff.is_on_staff_zone(bar_box):
                    bar_line_ids.append(f"b{k_bar}")
                    bar_line_staffs[k_bar].append(str(entry["id"]))
            entry["bar_line_ids"] = bar_line_ids
            for k_symbol, symbol in enumerate(staff.symbols):
                item: dict[str, object] = {
                    "id": f"{entry['id']}.{k_symbol}",
                    "kind": type(symbol).__name__,
                }
                box = getattr(symbol, "box", None)
                item["box"] = export.rotated(box)
                for attr in ("position", "has_dot"):
                    if hasattr(symbol, attr):
                        item[attr] = getattr(symbol, attr)
                stem = getattr(symbol, "stem", None)
                if stem is not None:
                    item["stem"] = export.rotated(stem)
                direction = getattr(symbol, "stem_direction", None)
                if direction is not None:
                    item["stem_direction"] = str(direction)
                symbols.append(item)
            entry["symbols"] = symbols
            staffs.append(entry)
    EXPORT.set_geometry(
        {
            "frame": "work",
            "staffs": staffs,
            "bar_lines": bar_lines,
            "canvas": [],
            "connections": [
                [export.rotated(c) for c in multi.connections] for multi in multi_staffs
            ],
            "unattached_notes": len(notes) if isinstance(notes, list) else None,
            "beams": {"status": "not_available"},
        },
    )


def predict_symbols(debug: Debug, predictions: InputPredictions) -> PredictedSymbols:
    eprint("Creating bounds for noteheads")
    noteheads = create_bounding_ellipses(predictions.notehead, min_size=(4, 4))
    eprint("Creating bounds for staff_fragments")
    staff_fragments = create_rotated_bounding_boxes(
        predictions.staff, skip_merging=True, min_size=(5, 1), max_size=(10000, 100)
    )

    eprint("Creating bounds for clefs_keys")
    clefs_keys = create_rotated_bounding_boxes(
        predictions.clefs_keys, min_size=(20, 40), max_size=(1000, 1000)
    )
    eprint("Creating bounds for stems_rest")
    stems_rest = create_rotated_bounding_boxes(predictions.stems_rest)
    eprint("Creating bounds for bar_lines")
    bar_line_img = prepare_bar_line_image(predictions.stems_rest)
    debug.write_threshold_image("bar_line_img", bar_line_img)
    bar_lines = create_rotated_bounding_boxes(bar_line_img, skip_merging=True, min_size=(1, 5))

    return PredictedSymbols(noteheads, staff_fragments, clefs_keys, stems_rest, bar_lines)


@dataclass
class ProcessingConfig:
    enable_debug: bool
    enable_cache: bool
    write_staff_positions: bool
    read_staff_positions: bool
    selected_staff: int
    # The transformer (encoder/decoder) only benefits from CUDA/ROCm: its fp16 "GPU"
    # models are slower than the fp32 ones when they end up on the CPU EP, and
    # the CoreML EP cannot run the decoder. Segnet additionally supports CoreML.
    transformer_use_gpu: bool
    segnet_use_gpu: bool
    # Opt-in (--coreml-encoder): run the encoder on the Apple GPU via CoreML.
    # Only helps across many images (slow one-time MLProgram compile).
    coreml_encoder: bool
    title_detection: bool


def process_image(
    image_path: str,
    config: ProcessingConfig,
    xml_generator_args: XmlGeneratorArguments,
) -> str:
    eprint("Processing " + image_path)
    xml_file = replace_extension(image_path, ".musicxml")
    debug_cleanup: Debug | None = None
    EXPORT.start_image(image_path, sys.argv[1:])
    exit_status = "exception"
    try:
        if config.read_staff_positions:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError("Failed to read " + image_path)
            input_size = (image.shape[1], image.shape[0])
            image = resize_image(image)
            EXPORT.frame("input", {"width": input_size[0], "height": input_size[1]})
            EXPORT.frame(
                "work",
                {
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "from_input": {
                        "crop": [0, 0, input_size[0], input_size[1]],
                        "scale": [image.shape[1] / input_size[0], image.shape[0] / input_size[1]],
                    },
                },
            )
            debug = Debug(image, image_path, config.enable_debug)
            staff_position_files = replace_extension(image_path, ".txt")
            multi_staffs = load_staff_positions(
                debug, image, staff_position_files, config.selected_staff
            )
            if EXPORT.on():
                EXPORT.grouping_update(
                    {
                        "source": "read_staff_positions",
                        "multi_staffs": [[EXPORT.staff_id(s) for s in m.staffs] for m in multi_staffs],
                        "staffs_out": _export_staffs([s for m in multi_staffs for s in m.staffs]),
                    }
                )
                _export_geometry(multi_staffs, [])
                EXPORT.stage("read_staff_positions", systems=len(multi_staffs))
            title = ""
            # parse_staffs() expects a grayscale image (cv2.findContours requires it),
            # matching what the normal detect_staffs_in_image() path hands it
            # (predictions.preprocessed). Apply the same CLAHE preprocessing here so the
            # two code paths feed the symbol-recognition encoder consistent input.
            image = color_adjust.apply_clahe(image)
        else:
            multi_staffs, image, debug, title_future, _ = detect_staffs_in_image(image_path, config)
        debug_cleanup = debug

        transformer_config = Config()
        transformer_config.use_gpu_inference = config.transformer_use_gpu
        transformer_config.use_coreml_encoder = config.coreml_encoder
        if EXPORT.on():
            paths = transformer_config.filepaths
            for name in ("encoder_path", "decoder_path", "encoder_path_fp16", "decoder_path_fp16"):
                path = getattr(paths, name, None)
                if isinstance(path, str):
                    EXPORT.model(name, path)
            EXPORT.model("segnet", segnet_path_onnx_fp16 if config.segnet_use_gpu else segnet_path_onnx)

        result_staffs = parse_staffs(
            debug,
            multi_staffs,
            image,
            selected_staff=config.selected_staff,
            config=transformer_config,
        )

        if not config.read_staff_positions:
            title = title_future.result(60)
        eprint("Found title:", title)

        EXPORT.stage("decode", voices=len(result_staffs))
        eprint("Writing XML", result_staffs)
        xml = generate_xml(xml_generator_args, result_staffs, title)
        EXPORT.stage("xml")
        ET.ElementTree(xml).write(xml_file, encoding="unicode", xml_declaration=True)

        eprint("Finished parsing " + str(len(result_staffs)) + " staves")
        teaser_file = replace_extension(image_path, "_teaser.png")
        if config.write_staff_positions:
            staff_position_files = replace_extension(image_path, ".txt")
            save_staff_positions(multi_staffs, image.shape, staff_position_files)
        debug.write_teaser(teaser_file, multi_staffs)
        debug.clean_debug_files_from_previous_runs()

        eprint("Result was written to", xml_file)
        exit_status = "ok"
    except BaseException as ex:
        exit_status = f"exception: {ex}"
        if os.path.exists(xml_file):
            os.remove(xml_file)
        raise
    finally:
        if debug_cleanup is not None:
            debug_cleanup.clean_debug_files_from_previous_runs()
        EXPORT.finish(exit_status)
    return xml_file


def detect_staffs_in_image(
    image_path: str, config: ProcessingConfig
) -> tuple[list[MultiStaff], NDArray, Debug, Future[str], int]:
    predictions, debug = load_and_preprocess_predictions(
        image_path, config.enable_debug, config.enable_cache, config.segnet_use_gpu
    )
    symbols = predict_symbols(debug, predictions)

    symbols.staff_fragments = break_wide_fragments(symbols.staff_fragments)
    debug.write_bounding_boxes("staff_fragments", symbols.staff_fragments)
    eprint("Found " + str(len(symbols.staff_fragments)) + " staff line fragments")

    noteheads_with_stems = combine_noteheads_with_stems(symbols.noteheads, symbols.stems_rest)
    debug.write_bounding_boxes_alternating_colors("notehead_with_stems", noteheads_with_stems)
    eprint("Found " + str(len(noteheads_with_stems)) + " noteheads")
    if len(noteheads_with_stems) == 0:
        _export_candidates(
            staff_fragments=symbols.staff_fragments,
            noteheads_with_stems=0,
            failure="No noteheads found",
        )
        raise Exception("No noteheads found")

    average_note_head_height = float(
        np.median([notehead.notehead.size[1] for notehead in noteheads_with_stems])
    )
    eprint("Average note head height: " + str(average_note_head_height))

    all_noteheads = [notehead.notehead for notehead in noteheads_with_stems]
    all_stems = [note.stem for note in noteheads_with_stems if note.stem is not None]
    bar_lines_or_rests = [
        line
        for line in symbols.bar_lines
        if not line.is_overlapping_with_any(all_noteheads)
        and not line.is_overlapping_with_any(all_stems)
    ]
    bar_line_boxes = detect_bar_lines(bar_lines_or_rests, average_note_head_height)
    debug.write_bounding_boxes_alternating_colors("bar_lines", bar_line_boxes)
    eprint("Found " + str(len(bar_line_boxes)) + " bar lines")

    debug.write_bounding_boxes(
        "anchor_input", symbols.staff_fragments + bar_line_boxes + symbols.clefs_keys
    )
    staffs = detect_staff(
        debug, predictions.staff, symbols.staff_fragments, symbols.clefs_keys, bar_line_boxes
    )
    _export_candidates(
        staff_fragments=symbols.staff_fragments,
        bar_line_boxes=bar_line_boxes,
        clefs_keys=symbols.clefs_keys,
        noteheads=[n.notehead for n in noteheads_with_stems],
        stems=all_stems,
        noteheads_with_stems=len(noteheads_with_stems),
        average_note_head_height=average_note_head_height,
        staffs_detected=_export_staffs(staffs) if EXPORT.on() else [],
        failure=None if len(staffs) > 0 else "No staffs found",
    )
    EXPORT.stage("staff_detection", staffs=len(staffs))
    if len(staffs) == 0:
        raise Exception("No staffs found")
    if config.title_detection:
        title_future = detect_title(debug, staffs[0])
    else:
        title_future = Future()
        title_future.set_result("")

    debug.write_bounding_boxes_alternating_colors("staffs", staffs)

    brace_dot_img = prepare_brace_dot_image(predictions.symbols, predictions.staff)
    debug.write_threshold_image("brace_dot", brace_dot_img)
    brace_dot = create_rotated_bounding_boxes(brace_dot_img, skip_merging=True, max_size=(100, -1))

    notes = add_notes_to_staffs(
        staffs, noteheads_with_stems, predictions.symbols, predictions.notehead
    )

    multi_staffs = find_braces_brackets_and_grand_staff_lines(debug, staffs, brace_dot)
    if EXPORT.on():
        EXPORT.grouping_update(
            {
                "staffs_in": [EXPORT.staff_id(s) for s in staffs],
                "brace_dot_boxes": [export.rotated(b) for b in brace_dot],
                "multi_staffs": [[EXPORT.staff_id(s) for s in m.staffs] for m in multi_staffs],
                "staffs_out": _export_staffs([s for m in multi_staffs for s in m.staffs]),
            }
        )
        _export_geometry(multi_staffs, notes, bar_line_boxes)
        EXPORT.stage("grouping", systems=len(multi_staffs))
    eprint(
        "Found",
        len(multi_staffs),
        "connected staffs (after merging grand staffs, multiple voices): ",
        [len(staff.staffs) for staff in multi_staffs],
    )

    debug.write_all_bounding_boxes_alternating_colors("notes", multi_staffs, notes)

    return multi_staffs, predictions.preprocessed, debug, title_future, len(staffs)


def get_all_image_files_in_folder(folder: str) -> list[str]:
    image_files = []
    for ext in ["png", "jpg", "jpeg", "pdf", "PNG", "JPG", "JPEG", "PDF"]:
        image_files.extend(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))
    without_teasers = [
        img
        for img in image_files
        if "_teaser" not in img
        and "_debug" not in img
        and "_staff" not in img
        and "_tesseract" not in img
    ]
    return sorted(without_teasers)


def download_weights(segnet_use_gpu: bool, transformer_use_gpu: bool, coreml_encoder: bool) -> None:
    base_url = "https://github.com/liebharc/homr/releases/download/onnx_checkpoints/"
    models = [segnet_path_onnx_fp16 if segnet_use_gpu else segnet_path_onnx]
    if transformer_use_gpu:
        # CUDA runs the whole transformer on the fp16 models.
        models.append(default_config.filepaths.encoder_path_fp16)
        models.append(default_config.filepaths.decoder_path_fp16)
    else:
        # On the CPU EP the fp32 models are faster, and the CoreML EP cannot run
        # the decoder, so the decoder always uses fp32. The CoreML encoder, when
        # enabled, uses the fp16 encoder instead of the fp32 one.
        if coreml_encoder:
            models.append(default_config.filepaths.encoder_path_fp16)
        else:
            models.append(default_config.filepaths.encoder_path)
        models.append(default_config.filepaths.decoder_path)
    missing_models = [model for model in models if not os.path.exists(model)]

    if len(missing_models) == 0:
        return

    eprint("Downloading", len(missing_models), "models - this is only required once")
    for model in missing_models:
        if not os.path.exists(model):
            base_name = os.path.basename(model).split(".")[0]
            eprint(f"Downloading {base_name}")
            try:
                zip_name = base_name + ".zip"
                download_url = base_url + zip_name
                downloaded_zip = os.path.join(os.path.dirname(model), zip_name)
                download_utils.download_file(download_url, downloaded_zip)

                destination_dir = os.path.dirname(model)
                download_utils.unzip_file(downloaded_zip, destination_dir)
            finally:
                if os.path.exists(downloaded_zip):
                    os.remove(downloaded_zip)


def run_homr(
    images: list, config: ProcessingConfig, xml_generator_args: XmlGeneratorArguments
) -> None:
    """
    Runs homr on all images in a list and merges them into musicxml using relieur
    """
    eprint("Merging", len(images), "files:", images)
    xml_paths = []
    filename = os.path.splitext(os.path.basename(images[0]))[0]
    for image_file in images:
        eprint("=========================================")
        try:
            xml_paths.append(process_image(image_file, config, xml_generator_args))
            eprint("Finished", image_file)
        except Exception as e:
            eprint(f"An error occurred while processing {image_file}: {e}")
            return  # Don't need to continue (save time)

    if len(xml_paths) == len(images) and len(xml_paths) > 1:
        output_path = os.path.join(root_dir, f"{filename}_merged.musicxml")
        m, _, _ = process_concat(xml_paths)
        ET.ElementTree(m).write(output_path, encoding="UTF-8", xml_declaration=True)

        for path in xml_paths:
            if os.path.exists(path):
                os.remove(path)

        eprint(f"Saved the generated musicxml at {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="homer", description="An optical music recognition (OMR) system"
    )
    parser.add_argument("image", type=str, nargs="*", help="Path to the image to process")
    parser.add_argument(
        "--init",
        action="store_true",
        help="Downloads the models if they are missing and then exits. "
        + "You don't have to call init before processing images, "
        + "it's only useful if you want to prepare for example a Docker image.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    parser.add_argument(
        "--cache", action="store_true", help="Read an existing cache file or create a new one"
    )
    parser.add_argument(
        "--output-large-page",
        action="store_true",
        help="Adds instructions to the musicxml so that it gets rendered on larger pages",
    )
    parser.add_argument(
        "--output-metronome", type=int, help="Adds a metronome to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--output-tempo", type=int, help="Adds a tempo to the musicxml with the given bpm"
    )
    parser.add_argument(
        "--write-staff-positions",
        action="store_true",
        help="Writes the position of all detected staffs to a txt file.",
    )
    parser.add_argument(
        "--read-staff-positions",
        action="store_true",
        help="Reads the position of all staffs from a txt file instead"
        + " of running the built-in staff detection.",
    )
    parser.add_argument(
        "--gpu",
        type=GpuSupport,
        choices=list(GpuSupport),
        default=GpuSupport.AUTO,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--coreml-encoder",
        action="store_true",
        help="On Apple Silicon, run the transformer encoder on the GPU via "
        + "CoreML. Compiling the model takes 26-60 s at startup, so this only "
        + "pays off when processing many images. Has no effect with CUDA.",
    )
    parser.add_argument(
        "--no-title", action="store_true", help="Don't detect title for faster inference"
    )
    parser.add_argument(
        "--export",
        type=str,
        default=None,
        metavar="HOOKS",
        help="Optionally export intermediate results next to the output, comma-separated: "
        + "topk, scores, scores-full, geometry, staffdetect, grouping or all. "
        + "Off by default; the MusicXML then only gains an id on every <note>.",
    )
    parser.add_argument(
        "--export-top-k",
        type=int,
        default=5,
        metavar="K",
        help="Candidates per decoder head for --export topk (default 5)",
    )
    parser.add_argument(
        "--export-dir", type=str, default=None, help="Directory for --export (default: next to the image)"
    )

    args = parser.parse_args()
    EXPORT.configure(args.export, args.export_top_k, args.export_dir)

    force_gpu = args.gpu == GpuSupport.FORCE
    auto_gpu = args.gpu == GpuSupport.AUTO

    # CUDA/ROCm speeds up the whole pipeline. CoreML only helps segnet: the fp16
    # models the GPU path uses are slower on the CPU EP than the fp32 ones,
    # and the CoreML EP cannot run the decoder (see Segnet for details).
    transformer_use_gpu = force_gpu or (auto_gpu and (cuda_available() or rocm_available()))
    segnet_use_gpu = force_gpu or (
        auto_gpu and (cuda_available() or rocm_available() or coreml_available())
    )
    # The CoreML encoder is a separate opt-in and only applies when the
    # transformer isn't already on CUDA/ROCm.
    coreml_encoder = args.coreml_encoder and not transformer_use_gpu and coreml_available()

    download_weights(segnet_use_gpu, transformer_use_gpu, coreml_encoder)

    if not args.no_title:
        download_ocr_weights()

    if args.init:
        eprint("Init finished")
        return

    config = ProcessingConfig(
        args.debug,
        args.cache,
        args.write_staff_positions,
        args.read_staff_positions,
        -1,
        transformer_use_gpu,
        segnet_use_gpu,
        coreml_encoder,
        not args.no_title,
    )

    xml_generator_args = XmlGeneratorArguments(
        args.output_large_page, args.output_metronome, args.output_tempo
    )
    if args.debug:
        eprint(f"Using Log Level {2} for OnnxRuntime")
        ort.set_default_logger_severity(2)
    else:
        ort.set_default_logger_severity(3)

    if not args.image:
        eprint("No image provided")
        parser.print_help()
        sys.exit(1)

    images_to_combine = []
    for image in args.image:
        if os.path.isfile(image):
            # Multiple files are getting merged
            if image.lower().endswith(".pdf"):
                images_to_combine += render_pdf_to_image(image)
            else:
                images_to_combine.append(image)

        elif os.path.isdir(image):
            # Directories are never merged
            image_files = get_all_image_files_in_folder(image)
            eprint("Processing", len(image_files), "files:", image_files)
            error_files = []
            for image_file in image_files:
                eprint("=========================================")
                try:
                    if image_file.lower().endswith(".pdf"):
                        rendered_images = render_pdf_to_image(image_file)
                        run_homr(rendered_images, config, xml_generator_args)
                    else:
                        process_image(image_file, config, xml_generator_args)
                    eprint("Finished", image_file)
                except Exception as e:
                    eprint(f"An error occurred while processing {image_file}: {e}")
                    error_files.append(image_file)
            if len(error_files) > 0:
                eprint("Errors occurred while processing the following files:", error_files)

        else:
            eprint(f"{image} is not a valid file or directory")
            sys.exit(2)

    # Check if we have any images
    if images_to_combine:
        # After collectiong all images we run homr on them
        try:
            run_homr(images_to_combine, config, xml_generator_args)
        except InvalidProgramArgumentException as e:
            eprint(str(e))
            sys.exit(2)


if __name__ == "__main__":
    main()
