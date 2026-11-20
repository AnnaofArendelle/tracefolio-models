# Inference preflight inputs

`ocr.png` is a Tracefolio-generated white 1000×400 PNG containing the public
text `TRACEFOLIO 2026`, drawn at (70,150) with Pillow and DejaVu Sans 72px.
It contains no source archive or personal data. The rendered pixels are
distributed under the project's AGPL-3.0-only license; no font file is bundled.

SHA-256: `7679eb1954c6e2eb09c6b78cbd8807f9b8d677382271bd4e06e103d724e3fd41`.

The supervisor stages this exact image and requires a real detection,
classification and recognition call returning `TRACEFOLIO` before acknowledging
readiness. This synthetic check does not establish Chinese recognition quality.

## Vision readiness input

`vision.jpg` retains the downloaded bytes of *Commander Eileen Collins -
GPN-2000-001177*, Robert Markowitz / NASA, published as public domain.
Source: https://commons.wikimedia.org/wiki/File:Commander_Eileen_Collins_-_GPN-2000-001177.jpg .
The source URL, original download hash and prior source record are copied into
`vision.source.json`. SHA256:
`86ff4d14156df3fa52e8fc101a11c1efa110bbcac79955533c64aeb7cc87e901`.
The model must detect a `person` and, when enabled, execute the pose network
before readiness. The person's published name is attribution for this photo,
not model output, a recognition target or an archive identity. This readiness
photo is not used as an accuracy benchmark.
