#!/usr/bin/env python3
"""
reussir_logo_generator.py

Generate Reussir programming language logos in three variants without matplotlib:
  1) Icon (hexagon + recycle emoji centered)
  2) High-res icon
  3) Wordmark banner (icon + text)

Rendering stack: Cairo + Pango (good color-emoji support with Noto Color Emoji).

Requirements (Ubuntu/Debian):
  sudo apt-get install -y python3-gi python3-cairo gir1.2-pango-1.0 \
                          fonts-noto fonts-noto-core fonts-noto-color-emoji
macOS (Homebrew):
  brew install pango cairo gobject-introspection pygobject3
  # Install Noto fonts (e.g., via homebrew-fonts or manual download)

Examples:
  python3 reussir_logo_generator.py --outdir out
  python3 reussir_logo_generator.py --palette "#4f46e5,#06b6d4" --outdir out

Notes:
- Ensure a color-emoji font is available (Noto Color Emoji recommended) so ♻️ renders in color.
- You can tweak fonts/colors/sizes with CLI options below.
"""

from __future__ import annotations
import math
import os
from pathlib import Path
import argparse

import cairo
from gi.repository import Pango, PangoCairo

# ----------------------------- Geometry Helpers ----------------------------- #

def hexagon_points(cx: float, cy: float, r: float, rotation_deg: float = 0.0):
    """Return 6 (x,y) vertices of a regular hexagon."""
    pts = []
    rot = math.radians(rotation_deg)
    for i in range(6):
        ang = rot + math.radians(60 * i - 30)  # flat-top style, like MLIR logo
        x = cx + r * math.cos(ang)
        y = cy + r * math.sin(ang)
        pts.append((x, y))
    return pts


def draw_polygon(ctx: cairo.Context, pts):
    ctx.new_path()
    x0, y0 = pts[0]
    ctx.move_to(x0, y0)
    for x, y in pts[1:]:
        ctx.line_to(x, y)
    ctx.close_path()


# ----------------------------- Drawing Primitives --------------------------- #

def fill_hexagon(ctx: cairo.Context, cx: float, cy: float, r: float,
                 color1: tuple[float, float, float],
                 color2: tuple[float, float, float],
                 rotation_deg: float = 0.0,
                 stroke_rgba: tuple[float, float, float, float] | None = (0, 0, 0, 0.08),
                 stroke_width: float = 4.0):
    pts = hexagon_points(cx, cy, r, rotation_deg)
    # gradient across bounding box
    x_min = min(p[0] for p in pts); x_max = max(p[0] for p in pts)
    y_min = min(p[1] for p in pts); y_max = max(p[1] for p in pts)
    grad = cairo.LinearGradient(x_min, y_min, x_max, y_max)
    grad.add_color_stop_rgb(0.0, *color1)
    grad.add_color_stop_rgb(1.0, *color2)

    draw_polygon(ctx, pts)
    ctx.set_source(grad)
    ctx.fill_preserve()
    if stroke_rgba is not None and stroke_width > 0:
        r, g, b, a = stroke_rgba
        ctx.set_source_rgba(r, g, b, a)
        ctx.set_line_width(stroke_width)
        ctx.stroke()
    else:
        ctx.new_path()


def pango_draw_text(ctx: cairo.Context, text: str, center_x: float, center_y: float,
                    font_desc: str, max_width: float | None = None,
                    color_rgba: tuple[float, float, float, float] | None = (1, 1, 1, 1),
                    markup: bool = False):
    """Draw centered text with Pango. If markup=True, 'text' is Pango markup."""
    layout = PangoCairo.create_layout(ctx)
    if markup:
        layout.set_markup(text, -1)
    else:
        layout.set_text(text, -1)
    fd = Pango.FontDescription(font_desc)
    layout.set_font_description(fd)
    if max_width is not None:
        layout.set_width(int(max_width * Pango.SCALE))
        layout.set_alignment(Pango.Alignment.CENTER)
    # Size in device pixels
    w, h = layout.get_pixel_size()
    ctx.save()
    if color_rgba is not None:
        ctx.set_source_rgba(*color_rgba)
    PangoCairo.update_layout(ctx, layout)
    PangoCairo.show_layout(ctx, layout)
    ctx.restore()
    return w, h


def draw_badge_emoji(ctx: cairo.Context, emoji: str, cx: float, cy: float,
                      badge_r: float, font_desc: str):
    # White badge circle with soft shadow
    ctx.save()
    # Shadow
    ctx.arc(cx + badge_r*0.08, cy + badge_r*0.08, badge_r, 0, 2*math.pi)
    ctx.set_source_rgba(0, 0, 0, 0.18)
    ctx.fill()
    # Core circle
    ctx.arc(cx, cy, badge_r, 0, 2*math.pi)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.fill()
    ctx.restore()

    # Emoji (let the color-emoji font render its own colors => color_rgba=None)
    pango_draw_text(ctx, emoji, cx, cy, font_desc=font_desc, color_rgba=None)


# ----------------------------- Icon Variants -------------------------------- #

def draw_icon(surface: cairo.Surface, size: int,
              palette=((0.31, 0.27, 0.90), (0.02, 0.71, 0.83)),
              lambda_font="Noto Sans, Noto Serif, DejaVu Sans 700 0",
              emoji_font="Noto Color Emoji 0",
              lambda_scale=0.58,
              badge_scale=0.2,
              emoji_scale: float = 1.20):
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    cx = cy = size / 2
    hex_r = size * 0.40

    # Background clear
    ctx.save()
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()
    ctx.restore()

    # Hexagon
    fill_hexagon(ctx, cx, cy, hex_r, color1=palette[0], color2=palette[1], rotation_deg=0)

    # Centered recycle emoji inside the hexagon (no lambda, no badge)
    # Use a slightly larger-than-radius pixel size to read well while staying centered
    emoji_px = int(hex_r * emoji_scale)
    emoji_desc = f"{emoji_font} {emoji_px}px"
    pango_draw_text(ctx, "♻️", cx, cy, emoji_desc, color_rgba=None)


# ----------------------------- Banner Variant -------------------------------- #

def draw_banner(surface: cairo.Surface, width: int, height: int,
                palette=((0.31, 0.27, 0.90), (0.02, 0.71, 0.83)),
                wordmark_font="Noto Sans Display, Noto Sans, Inter 900 0",
                tagline_font="Noto Sans, Inter 400 0",
                emoji_font="Noto Color Emoji 0"):
    ctx = cairo.Context(surface)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)

    # Background
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.paint()

    # Left icon
    icon_size = int(min(height * 0.82, width * 0.32))
    icon_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, icon_size, icon_size)
    draw_icon(icon_surface, icon_size, palette=palette)
    # composite icon
    x_icon = int(height * 0.09)
    y_icon = int((height - icon_size) / 2)
    ctx.set_source_surface(icon_surface, x_icon, y_icon)
    ctx.paint()

    # Wordmark (right side)
    right_margin = int(width * 0.06)
    left_text = x_icon + icon_size + int(width * 0.04)
    text_width = width - left_text - right_margin

    # Title: Reussir Programming Language (single line, centered vertically)
    title = "Reussir Programming Language"
    layout = PangoCairo.create_layout(ctx)

    # Start with a large size, then downscale until it fits the available width
    title_px = int(height * 0.24)
    layout.set_text(title, -1)
    layout.set_alignment(Pango.Alignment.LEFT)
    layout.set_width(-1)  # no wrapping

    for _ in range(40):  # guard loop
        layout.set_font_description(Pango.FontDescription(f"{wordmark_font} {title_px}px"))
        tw, th = layout.get_pixel_size()
        if tw <= text_width or title_px <= 10:
            break
        title_px = max(10, int(title_px * 0.96))

    # Position: left aligned in text area, vertically centered
    tw, th = layout.get_pixel_size()
    tx = left_text
    ty = int((height - th) / 2)

    # Create text path and fill with gradient
    ctx.save()
    ctx.translate(tx, ty)
    PangoCairo.layout_path(ctx, layout)
    grad = cairo.LinearGradient(tx, ty, tx + max(tw, 1), ty + th)
    c1, c2 = palette
    grad.add_color_stop_rgb(0.0, *c1)
    grad.add_color_stop_rgb(1.0, *c2)
    ctx.set_source(grad)
    ctx.fill_preserve()
    # Subtle stroke
    ctx.set_source_rgba(0, 0, 0, 0.08)
    ctx.set_line_width(1.0)
    ctx.stroke()
    ctx.restore()

    # No tagline — per request keep it plain


# ----------------------------- Surfaces & IO --------------------------------- #

def hex_to_rgb(hex_str: str) -> tuple[float, float, float]:
    hex_str = hex_str.strip().lstrip('#')
    if len(hex_str) == 3:
        r, g, b = (int(hex_str[i]*2, 16) for i in range(3))
    elif len(hex_str) == 6:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
    else:
        raise ValueError(f"Invalid hex color: {hex_str}")
    return (r/255.0, g/255.0, b/255.0)


def make_png_and_svg(basepath: Path, draw_fn, *args, **kwargs):
    # SVG
    svg_path = basepath.with_suffix('.svg')
    png_path = basepath.with_suffix('.png')
    if 'width' in kwargs and 'height' in kwargs:
        W, H = kwargs['width'], kwargs['height']
    elif 'size' in kwargs:
        W = H = kwargs['size']
    else:
        raise ValueError("Provide width/height or size")

    # SVG surface
    with cairo.SVGSurface(str(svg_path), W, H) as svg_surf:
        draw_fn(svg_surf, *args, **kwargs)

    # PNG surface
    img_surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, W, H)
    draw_fn(img_surf, *args, **kwargs)
    img_surf.write_to_png(str(png_path))

    return png_path, svg_path


# ----------------------------- CLI ------------------------------------------ #

def parse_args():
    ap = argparse.ArgumentParser(description="Generate Reussir logos (icon/highres/banner) with Cairo + Pango.")
    ap.add_argument('--outdir', type=Path, default=Path('logos'), help='Output directory')
    ap.add_argument('--basename', type=str, default='reussir', help='Base filename')
    ap.add_argument('--icon-size', type=int, default=512, help='Icon size (square)')
    ap.add_argument('--hires-size', type=int, default=2048, help='High-res icon size (square)')
    ap.add_argument('--banner-size', type=str, default='1600x800', help='Banner WxH')
    ap.add_argument('--palette', type=str, default='#4f46e5,#06b6d4', help='Two hex colors, comma-separated')
    ap.add_argument('--lambda-font', type=str, default='Noto Sans, Noto Serif, DejaVu Sans 700 0', help='Deprecated: kept for compatibility (unused)')
    ap.add_argument('--emoji-font', type=str, default='Noto Color Emoji 0', help='Emoji font (family)')
    ap.add_argument('--wordmark-font', type=str, default='Noto Sans Display, Noto Sans, Inter 900 0', help='Font stack for title')
    ap.add_argument('--tagline-font', type=str, default='Noto Sans, Inter 400 0', help='Deprecated: kept for compatibility (unused)')
    ap.add_argument('--emoji-scale', type=float, default=1.20, help='Scale factor for emoji size relative to hex radius')
    return ap.parse_args()


def main():
    args = parse_args()
    outdir = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    c1_hex, c2_hex = [s.strip() for s in args.palette.split(',')]
    palette = (hex_to_rgb(c1_hex), hex_to_rgb(c2_hex))

    # Icon
    icon_base = outdir / f"{args.basename}_icon_{args.icon_size}"
    make_png_and_svg(
        icon_base,
        draw_icon,
        size=args.icon_size,
        palette=palette,
        lambda_font=args.lambda_font,
        emoji_font=args.emoji_font,
        emoji_scale=args.emoji_scale,
    )

    # High-res icon
    hires_base = outdir / f"{args.basename}_icon_{args.hires_size}"
    make_png_and_svg(
        hires_base,
        draw_icon,
        size=args.hires_size,
        palette=palette,
        lambda_font=args.lambda_font,
        emoji_font=args.emoji_font,
        emoji_scale=args.emoji_scale,
    )

    # Banner (wordmark)
    try:
        bw, bh = map(int, args.banner_size.lower().split('x'))
    except Exception:
        raise SystemExit("--banner-size must be like WIDTHxHEIGHT, e.g. 1600x800")

    banner_base = outdir / f"{args.basename}_banner_{bw}x{bh}"
    make_png_and_svg(
        banner_base,
        draw_banner,
        width=bw,
        height=bh,
        palette=palette,
        wordmark_font=args.wordmark_font,
        tagline_font=args.tagline_font,
        emoji_font=args.emoji_font,
    )

    print(f"Wrote: {outdir}")


if __name__ == '__main__':
    main()
