"""A 5x7 bitmap font, for rendering plate text with numpy alone.

The synthetic source has to draw plates that a plate detector can find and an OCR
stage can attempt. Doing that with cv2.putText would make the one adapter whose
entire purpose is to run anywhere depend on OpenCV, and the tests that exist
precisely because the CI machine has no GPU and no video codecs would stop
running there.

So: the glyphs are data. Thirty-seven 5x7 masks, written as text so they are
verifiable by reading them. Nobody has to trust that this file is correct -- the
letters are visible in the source.

Not a general-purpose renderer. Upper-case letters, digits and space, which is
exactly the character set Contracts section 4.2 normalizes plates into.
"""

from typing import Optional

import numpy as np

GLYPH_WIDTH = 5
GLYPH_HEIGHT = 7

_GLYPH_SOURCE: dict[str, str] = {
    "0": ".###." "#...#" "#...#" "#...#" "#...#" "#...#" ".###.",
    "1": "..#.." ".##.." "..#.." "..#.." "..#.." "..#.." ".###.",
    "2": ".###." "#...#" "....#" "...#." "..#.." ".#..." "#####",
    "3": ".###." "#...#" "....#" "..##." "....#" "#...#" ".###.",
    "4": "...#." "..##." ".#.#." "#..#." "#####" "...#." "...#.",
    "5": "#####" "#...." "####." "....#" "....#" "#...#" ".###.",
    "6": "..##." ".#..." "#...." "####." "#...#" "#...#" ".###.",
    "7": "#####" "....#" "...#." "..#.." ".#..." ".#..." ".#...",
    "8": ".###." "#...#" "#...#" ".###." "#...#" "#...#" ".###.",
    "9": ".###." "#...#" "#...#" ".####" "....#" "...#." ".##..",
    "A": "..#.." ".#.#." "#...#" "#...#" "#####" "#...#" "#...#",
    "B": "####." "#...#" "#...#" "####." "#...#" "#...#" "####.",
    "C": ".###." "#...#" "#...." "#...." "#...." "#...#" ".###.",
    "D": "####." "#...#" "#...#" "#...#" "#...#" "#...#" "####.",
    "E": "#####" "#...." "#...." "####." "#...." "#...." "#####",
    "F": "#####" "#...." "#...." "####." "#...." "#...." "#....",
    "G": ".###." "#...#" "#...." "#.###" "#...#" "#...#" ".###.",
    "H": "#...#" "#...#" "#...#" "#####" "#...#" "#...#" "#...#",
    "I": ".###." "..#.." "..#.." "..#.." "..#.." "..#.." ".###.",
    "J": "..###" "...#." "...#." "...#." "...#." "#..#." ".##..",
    "K": "#...#" "#..#." "#.#.." "##..." "#.#.." "#..#." "#...#",
    "L": "#...." "#...." "#...." "#...." "#...." "#...." "#####",
    "M": "#...#" "##.##" "#.#.#" "#...#" "#...#" "#...#" "#...#",
    "N": "#...#" "##..#" "#.#.#" "#..##" "#...#" "#...#" "#...#",
    "O": ".###." "#...#" "#...#" "#...#" "#...#" "#...#" ".###.",
    "P": "####." "#...#" "#...#" "####." "#...." "#...." "#....",
    "Q": ".###." "#...#" "#...#" "#...#" "#.#.#" "#..#." ".##.#",
    "R": "####." "#...#" "#...#" "####." "#.#.." "#..#." "#...#",
    "S": ".###." "#...#" "#...." ".###." "....#" "#...#" ".###.",
    "T": "#####" "..#.." "..#.." "..#.." "..#.." "..#.." "..#..",
    "U": "#...#" "#...#" "#...#" "#...#" "#...#" "#...#" ".###.",
    "V": "#...#" "#...#" "#...#" "#...#" "#...#" ".#.#." "..#..",
    "W": "#...#" "#...#" "#...#" "#.#.#" "#.#.#" "##.##" "#...#",
    "X": "#...#" "#...#" ".#.#." "..#.." ".#.#." "#...#" "#...#",
    "Y": "#...#" "#...#" ".#.#." "..#.." "..#.." "..#.." "..#..",
    "Z": "#####" "....#" "...#." "..#.." ".#..." "#...." "#####",
    " ": "....." "....." "....." "....." "....." "....." ".....",
}


def _compile(source: str) -> np.ndarray:
    expected = GLYPH_WIDTH * GLYPH_HEIGHT
    if len(source) != expected:
        raise ValueError(f"glyph must be {expected} cells, got {len(source)}")
    flat = np.frombuffer(source.encode("ascii"), dtype=np.uint8) == ord("#")
    return flat.reshape(GLYPH_HEIGHT, GLYPH_WIDTH)


GLYPHS: dict[str, np.ndarray] = {ch: _compile(src) for ch, src in _GLYPH_SOURCE.items()}

# Fallback for any character outside the set: a filled block. Visible as wrong
# rather than invisible as missing, so a bad plate string in a fixture shows up
# on the image instead of producing a silently shorter plate.
_UNKNOWN = np.ones((GLYPH_HEIGHT, GLYPH_WIDTH), dtype=bool)


def text_mask(text: str, *, scale: int = 1, spacing: int = 1) -> np.ndarray:
    """Render text to a boolean mask, True where ink is.

    scale is nearest-neighbour integer upscaling. Nearest-neighbour and not
    smoothing: a synthetic plate should be exactly as sharp as its scale implies,
    so that a quality score computed from it is a property of the declared size
    rather than of a resampling filter.
    """
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    if not text:
        return np.zeros((GLYPH_HEIGHT * scale, 0), dtype=bool)

    columns = []
    for index, char in enumerate(text.upper()):
        if index:
            columns.append(np.zeros((GLYPH_HEIGHT, spacing), dtype=bool))
        columns.append(GLYPHS.get(char, _UNKNOWN))

    mask = np.concatenate(columns, axis=1)
    if scale > 1:
        mask = np.repeat(np.repeat(mask, scale, axis=0), scale, axis=1)
    return mask


def text_extent(text: str, *, scale: int = 1, spacing: int = 1) -> tuple[int, int]:
    """(width, height) in pixels for the given text, without rendering it."""
    if not text:
        return (0, GLYPH_HEIGHT * scale)
    width = len(text) * GLYPH_WIDTH + (len(text) - 1) * spacing
    return (width * scale, GLYPH_HEIGHT * scale)


def draw_text(
    canvas: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    *,
    scale: int = 1,
    spacing: int = 1,
) -> Optional[tuple[int, int, int, int]]:
    """Draw text onto a BGR canvas in place. Returns the xyxy box actually drawn.

    Returns None when the text falls entirely outside the canvas. Clipping rather
    than raising: a vehicle driving off the edge of the frame is normal, and a
    generator that crashes at the edge of the road is useless for the boundary
    cases that are exactly what needs testing.
    """
    mask = text_mask(text, scale=scale, spacing=spacing)
    if mask.size == 0:
        return None

    x0, y0 = origin
    height, width = mask.shape
    canvas_h, canvas_w = canvas.shape[:2]

    src_x0 = max(0, -x0)
    src_y0 = max(0, -y0)
    dst_x0 = max(0, x0)
    dst_y0 = max(0, y0)
    copy_w = min(width - src_x0, canvas_w - dst_x0)
    copy_h = min(height - src_y0, canvas_h - dst_y0)
    if copy_w <= 0 or copy_h <= 0:
        return None

    window = mask[src_y0 : src_y0 + copy_h, src_x0 : src_x0 + copy_w]
    region = canvas[dst_y0 : dst_y0 + copy_h, dst_x0 : dst_x0 + copy_w]
    region[window] = np.array(colour, dtype=canvas.dtype)

    return (dst_x0, dst_y0, dst_x0 + copy_w, dst_y0 + copy_h)
