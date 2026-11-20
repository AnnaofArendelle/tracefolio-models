# SPDX-License-Identifier: AGPL-3.0-only
import io

import pypdfium2 as pdf
import pytest
from PIL import Image

from tracefolio_worker.description import caption_image, caption_page
from tracefolio_worker.documents.pdf import probe_pdf


def test_caption_image_orientation_crop_and_small_input(tmp_path):
    source = tmp_path / "sample.png"
    image = Image.new("RGB", (1200, 600), "red")
    image.paste("blue", (600, 0, 1200, 600))
    image.save(source)
    with Image.open(io.BytesIO(caption_image(source, [0, 0, 600, 600], 3))) as result:
        assert result.size == (384, 384)
        assert result.getpixel((192, 192))[2] > 240
    with Image.open(io.BytesIO(caption_image(source))) as result:
        assert result.size == (384, 192)
    with pytest.raises(ValueError, match="outside"):
        caption_image(source, [0, 0, 1500, 600])


@pytest.mark.parametrize("rotation", [0, 90, 180, 270])
def test_caption_pdf_page_and_region_are_bounded(rotation):
    with pdf.PdfDocument(probe_pdf()) as document:
        page = document[0]
        page.set_rotation(rotation)
        try:
            with Image.open(io.BytesIO(caption_page(page))) as result:
                assert max(result.size) == 384
            with Image.open(
                io.BytesIO(
                    caption_page(
                        page,
                        {
                            "left": 10,
                            "bottom": 10,
                            "right": 110,
                            "top": 50,
                        },
                    )
                )
            ) as result:
                assert max(result.size) <= 384
        finally:
            page.close()
