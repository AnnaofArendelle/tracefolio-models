# SPDX-License-Identifier: AGPL-3.0-only
"""Apply an explicit direction without altering or trusting original EXIF."""


def oriented_image(source, orientation):
    from PIL import Image

    if type(orientation) is not int or not 1 <= orientation <= 8:
        raise ValueError("image direction requires confirmation")
    transforms = {
        2: Image.Transpose.FLIP_LEFT_RIGHT,
        3: Image.Transpose.ROTATE_180,
        4: Image.Transpose.FLIP_TOP_BOTTOM,
        5: Image.Transpose.TRANSPOSE,
        6: Image.Transpose.ROTATE_270,
        7: Image.Transpose.TRANSVERSE,
        8: Image.Transpose.ROTATE_90,
    }
    return source.transpose(transforms[orientation]) if orientation in transforms else source.copy()
