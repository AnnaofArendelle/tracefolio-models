# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed adapters only. No provider, download, user code or shell execution."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tracefolio_worker.ocr.rapid import Asset, OCRModels, OCRThresholds, RapidAdapter
from tracefolio_worker.protocol.wire import staged_file
from tracefolio_worker.ocr.correction import validate_correction


def checked_ocr(models, assets, probe, sentinel):
    if "ocr" not in models:
        adapter = RapidAdapter(
            OCRModels(assets["det"], assets["rec"], assets["cls"], assets["dictionary"])
        )
        models["ocr"] = adapter, adapter.preflight(probe, sentinel)
    return models["ocr"]


def dispatch(req, wire, models=None, resources=None):
    if models is None:
        models = {}
    config = req["input"]
    if req["module"] in {"description", "description-render"}:
        from tracefolio_worker.description import run_description

        return run_description(
            config, wire, req["module"] == "description-render", models, resources
        )
    required = {"runtime", "staging", "source", "probe", "sentinel", "next_ordinal"}
    if req["module"] == "ocr":
        required |= {"input_schema", "crop"}
        if config.get("input_schema") != 2 or type(config.get("input_schema")) is not int:
            raise ValueError("unsupported OCR input schema")
        if "units" in config:
            required.add("units")
            if not isinstance(config["units"], list) or not 1 <= len(config["units"]) <= 1280:
                raise ValueError("invalid OCR operation units")
            for unit in config["units"]:
                if (
                    not isinstance(unit, dict)
                    or set(unit) - {"correction", "corrected_crop", "thresholds"}
                    != {"crop", "depth", "single_line", "tile"}
                    or unit["depth"] not in {"fast", "standard"}
                    or type(unit["single_line"]) is not bool
                    or type(unit["tile"]) is not bool
                ):
                    raise ValueError("invalid OCR unit")
                validate_correction(unit.get("correction"))
                unit["thresholds"] = OCRThresholds.from_request(unit.get("thresholds"))
                window = unit.get("corrected_crop")
                if window is not None and (
                    unit.get("correction") is None
                    or not isinstance(window, dict)
                    or set(window) != {"left", "top", "right", "bottom"}
                    or any(type(v) is not int for v in window.values())
                ):
                    raise ValueError("invalid corrected tile")
                if unit["crop"] is not None and (
                    not isinstance(unit["crop"], dict)
                    or set(unit["crop"]) != {"left", "top", "right", "bottom"}
                ):
                    raise ValueError("invalid OCR unit crop")
    if req["module"] in {"objects", "pose"}:
        required |= {"input_schema", "crop"}
        if type(config.get("input_schema")) is not int or config["input_schema"] != 2:
            raise ValueError("unsupported visual input schema")
    if req["module"] == "pdf":
        required |= {"input_schema", "first_page", "last_page", "pdf_region"}
        if type(config.get("input_schema")) is not int or config["input_schema"] != 2:
            raise ValueError("unsupported PDF input schema")
        if (
            type(config["first_page"]) is not int
            or type(config["last_page"]) is not int
            or config["first_page"] < 1
            or config["last_page"] < 0
        ):
            raise ValueError("invalid PDF page selection")
        if "pdf_regions" in config:
            required |= {"pdf_regions", "next_region"}
            regions = config["pdf_regions"]
            if (
                regions is not None
                and (not isinstance(regions, list) or not 1 <= len(regions) <= 128)
                or type(config["next_region"]) is not int
                or not 0 <= config["next_region"] < (len(regions) if regions else 1)
                or regions
                and (
                    config["pdf_region"] is not None or config["first_page"] != config["last_page"]
                )
            ):
                raise ValueError("invalid PDF batch regions")
        if "options" in config:
            required |= {"options", "next_unit"}
            options = config["options"]
            if (
                not isinstance(options, dict)
                or set(options) - {"correction", "thresholds"}
                != {"depth", "single_line", "tile_start"}
                or options["depth"] not in {"fast", "standard", "detailed"}
                or type(options["single_line"]) is not bool
                or type(options["tile_start"]) is not int
                or not 0 <= options["tile_start"] <= 1000000
                or type(config["next_unit"]) is not int
                or not 0 <= config["next_unit"] <= 10
            ):
                raise ValueError("invalid PDF operation options")
            validate_correction(options.get("correction"))
            options["thresholds"] = OCRThresholds.from_request(options.get("thresholds"))
    if req["module"] == "asr" and config.get("input_schema") == 2:
        required |= {"input_schema", "next_clock"}
    if req["module"] == "asr" and config.get("input_schema") == 3:
        required |= {"input_schema", "review", "media_origin_ns"}
    if req["module"] == "video":
        required |= {
            "input_schema",
            "vision_probe",
            "with_ocr",
            "with_objects",
            "with_pose",
            "region",
        }
        if type(config.get("input_schema")) is not int or config["input_schema"] != 1:
            raise ValueError("unsupported video input schema")
    if "orientation_override" in config:
        if (
            req["module"] not in {"ocr", "objects", "pose"}
            or type(config["orientation_override"]) is not int
            or not 1 <= config["orientation_override"] <= 8
        ):
            raise ValueError("invalid confirmed image direction")
        required.add("orientation_override")
    if set(config) != required:
        raise ValueError("unsupported adapter configuration")
    if type(config["next_ordinal"]) is not int or config["next_ordinal"] < 0:
        raise ValueError("invalid checkpoint")
    runtime, staging = Path(config["runtime"]), Path(config["staging"])
    if not runtime.is_absolute() or not staging.is_absolute():
        raise ValueError("supervisor roots must be absolute")
    source, probe = staged_file(staging, config["source"]), staged_file(staging, config["probe"])
    if "assets" not in models:
        manifest = json.loads((runtime / "models.json").read_text())
        assets = {}
        for item in manifest["assets"]:
            path = Path(item["path"])
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("unsafe model path")
            assets[item["role"]] = Asset(runtime / path, item["sha256"])
        models["assets"] = assets
    assets = models["assets"]
    module = req["module"]
    sentinel = config["sentinel"]
    if not isinstance(sentinel, str) or not 1 <= len(sentinel) <= 256:
        raise ValueError("invalid preflight sentinel")
    start = config["next_ordinal"]

    if module == "asr" and config.get("input_schema") == 2:
        from tracefolio_worker.asr.media import run_asr

        return run_asr(config, wire, assets, source, probe, models)

    if module == "asr" and config.get("input_schema") == 3:
        from tracefolio_worker.asr.media import run_asr_review

        return run_asr_review(config, wire, assets, source, probe, models)

    if module == "video":
        from tracefolio_worker.media.video import run_video

        return run_video(config, wire, assets, source, probe, runtime, staging, models)

    def ready(proof):
        wire.emit(
            "ready", {"module": module, "probe_sha256": config["probe"]["sha256"], "proof": proof}
        )

    def artifact(ordinal, result):
        wire.emit(
            "artifact",
            {
                "module": module,
                "ordinal": ordinal,
                "result": result,
                "checkpoint": {"next_ordinal": ordinal + 1},
            },
        )

    if module == "ocr":
        adapter, result = checked_ocr(models, assets, probe, sentinel)
        ready({"sentinel": sentinel, "segments": len(result["segments"])})
        if "units" in config:
            if start > len(config["units"]):
                raise ValueError("OCR checkpoint outside operation")
            for index in range(start, len(config["units"])):
                unit = config["units"][index]
                crop = unit["crop"]
                bounds = [crop[k] for k in ("left", "top", "right", "bottom")] if crop else None
                artifact(
                    index,
                    adapter.extract(
                        source,
                        bounds,
                        config.get("orientation_override"),
                        unit["depth"],
                        unit["single_line"],
                        correction=unit.get("correction"),
                        corrected_crop=unit.get("corrected_crop"),
                        thresholds=unit["thresholds"],
                    ),
                )
        elif start == 0:
            artifact(0, adapter.extract(source, config["crop"], config.get("orientation_override")))
    elif module == "pdf":
        import pypdfium2 as pdf
        from tracefolio_worker.documents.pdf import MAX_PAGES, PDFAdapter, verify_pdfium

        if "pdfium" not in models:
            verify_pdfium()
            models["pdfium"] = True
        with pdf.PdfDocument(source) as document:
            total = len(document)
            first = config["first_page"] - 1
            stop = (
                min(total, MAX_PAGES)
                if config["last_page"] == 0
                else min(total, config["last_page"])
            )
            if first >= total or stop <= first or start > stop:
                raise ValueError("PDF checkpoint/page range beyond document")
            ready({"native_sentinel": "TRACEFOLIO PDF", "total_pages": total})

            def make_ocr():
                return checked_ocr(models, assets, probe, sentinel)[0]

            adapter = PDFAdapter(make_ocr)
            for number in range(max(start, first), stop):
                page = document[number]
                try:
                    if "options" not in config:
                        artifact(
                            number,
                            adapter.page(page, number + 1, total, staging, config["pdf_region"]),
                        )
                    else:
                        options = config["options"]
                        selected_regions = config.get("pdf_regions") or [config["pdf_region"]]
                        region_start = config.get("next_region", 0) if number == start else 0
                        for region_index in range(region_start, len(selected_regions)):
                            selected_region = selected_regions[region_index]
                            units, next_tile, total_tiles = adapter.units(
                                page, selected_region, options
                            )
                            unit_start = (
                                config["next_unit"]
                                if number == start and region_index == region_start
                                else 0
                            )
                            if unit_start >= len(units):
                                raise ValueError("PDF unit checkpoint beyond page")
                            for unit_index in range(unit_start, len(units)):
                                region, scale, raster_region, supplement = units[unit_index]
                                result = adapter.page(
                                    page,
                                    number + 1,
                                    total,
                                    staging,
                                    region,
                                    scale=scale,
                                    depth="fast" if options["depth"] == "fast" else "standard",
                                    single_line=options["single_line"],
                                    raster_region=raster_region,
                                    correction=options.get("correction"),
                                    thresholds=options["thresholds"],
                                )
                                result["unit_index"] = unit_index
                                if config.get("pdf_regions"):
                                    result["region_index"] = region_index
                                result["thresholds"] = asdict(options["thresholds"])
                                result["supplement"] = supplement
                                if options["depth"] == "detailed":
                                    completed = (
                                        options["tile_start"]
                                        + unit_index
                                        + (1 if options["tile_start"] else 0)
                                    )
                                    result["detail"] = {
                                        **(
                                            {"correction": options["correction"]}
                                            if options.get("correction") is not None
                                            else {}
                                        ),
                                        "region": selected_region,
                                        "thresholds": asdict(options["thresholds"]),
                                        "next_tile": completed,
                                        "total_tiles": total_tiles,
                                    }
                                last_unit = unit_index == len(units) - 1
                                last_region = region_index == len(selected_regions) - 1
                                wire.emit(
                                    "artifact",
                                    {
                                        "module": "pdf",
                                        "ordinal": number,
                                        "result": result,
                                        "checkpoint": {
                                            "next_ordinal": number + 1
                                            if last_unit and last_region
                                            else number,
                                            **(
                                                {
                                                    "next_region": 0
                                                    if last_unit and last_region
                                                    else region_index + (1 if last_unit else 0)
                                                }
                                                if config.get("pdf_regions")
                                                else {}
                                            ),
                                            "next_unit": 0 if last_unit else unit_index + 1,
                                        },
                                    },
                                )
                finally:
                    page.close()
            wire.emit(
                "done",
                {
                    "status": "completed",
                    "total_pages": total,
                    "processed_pages": stop,
                },
            )
        return
    elif module in {"objects", "pose"}:
        from tracefolio_worker.vision.objects import ObjectAdapter, load_image
        from tracefolio_worker.vision.pose import PoseAdapter

        if "objects" not in models:
            detector = ObjectAdapter(assets["object"])
            models["objects"] = detector, detector.preflight(probe, sentinel)
        detector, detected = models["objects"]
        result = None
        pixels = None
        people = []
        if start == 0:
            pixels, orientation = load_image(
                source, config["crop"], config.get("orientation_override")
            )
            result = detector.extract_pixels(pixels)
            result["exif_orientation"] = orientation
            people = [o for o in result["objects"] if o["label"] == "person"]
        pose = None
        # RTMPose is human-only. Empty scenes and animal/object-only frames do
        # not construct a pose session or run its probe; use this detection once.
        if module == "pose" and people:
            if "pose" not in models:
                pose = PoseAdapter(assets["pose"])
                probe_people = [o for o in detected["objects"] if o["label"] == "person"]
                if not probe_people:
                    raise ValueError("pose preflight needs an actual detected person")
                probe_pixels, _ = load_image(probe)
                posed = pose.extract_pixels(probe_pixels, probe_people[0])
                if len(posed["points"]) != 17:
                    raise ValueError("pose inference preflight failed")
                models["pose"] = pose
            pose = models["pose"]
        ready(
            {
                "label": sentinel,
                "detections": len(detected["objects"]),
                "pose_inferred": pose is not None,
            }
        )
        if start == 0:
            if module == "pose":
                result["poses"] = [pose.extract_pixels(pixels, o) for o in people]
            artifact(0, result)
    else:
        import wave

        from tracefolio_worker.asr.sensevoice import MAX_SAMPLES, SenseVoiceAdapter, bounded_chunks
        from tracefolio_worker.asr.vad import SileroAdapter

        vad = SileroAdapter(assets["vad"])
        asr = SenseVoiceAdapter(assets["asr"], assets["tokens"])
        # Probe is a bounded known sample; actual recordings always stream.
        with wave.open(str(probe)) as audio:
            length = audio.getnframes()
        if not 0 < length <= MAX_SAMPLES:
            raise ValueError("ASR preflight must be a bounded sample")
        ranges = list(vad.ranges(probe))
        asr.preflight(probe, 0, length, sentinel)
        if not ranges:
            raise ValueError("VAD preflight sentinel produced no speech")
        ready({"sentinel": sentinel, "vad_speech_ranges": len(ranges)})
        ordinal = 0
        for first, last in vad.ranges(source):
            for a, b in bounded_chunks(first, last):
                if ordinal >= start:
                    artifact(ordinal, asr.extract(source, a, b))
                ordinal += 1
        if start > ordinal:
            raise ValueError("checkpoint beyond deterministic audio blocks")
    if module != "asr" and "units" not in config and start > 1:
        raise ValueError("checkpoint outside single image")
    wire.emit("done", {"status": "completed"})
