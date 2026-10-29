# -*- coding: utf-8 -*-
"""
Remove watermarks from PDF files.

Two watermark types are supported:

1. Vector/tiled watermark: a faint image tiled with a PDF tiling pattern
   (PatternType 1).  Such tiles usually carry an /SMask with low alpha.
2. Baked-in screenshot watermark: a faint, periodically repeated text overlay
   that is part of an embedded raster image.  Enabled with --clean-baked.

Strategy for (1): locate tiling patterns that paint a low-alpha image, blank
those pattern streams and drop their resource links.  Text, tables, fonts and
vector graphics stay untouched.  The input file is never modified.

Strategy for (2): detect the repeating lattice of the faint overlay inside an
embedded image, reconstruct the watermark layer from all repetitions and invert
the alpha blend in place.  Pixels over plain background are filled from the
local background; pixels over text/lines are restored by inverting the blend.

Usage:
    python remove_pdf_watermark.py input.pdf
    python remove_pdf_watermark.py input.pdf --clean-baked
    python remove_pdf_watermark.py input.pdf -o output.pdf
    python remove_pdf_watermark.py input.pdf --dry-run
    python remove_pdf_watermark.py input.pdf --pattern 82,83
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pymupdf


@dataclass
class Candidate:
    pattern_xref: int
    image_xref: int
    image_width: int
    image_height: int
    smask_max: int
    smask_mean: float
    image_refs: list = field(default_factory=list)


def _xref_int(doc, xref, key, default=None):
    kind, value = doc.xref_get_key(xref, key)
    if kind in ("int", "float"):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _xref_float(doc, xref, key, default=None):
    kind, value = doc.xref_get_key(xref, key)
    if kind in ("int", "float"):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _resources_text(doc, xref):
    kind, value = doc.xref_get_key(xref, "Resources")
    if kind == "dict":
        return value
    if kind == "xref":
        try:
            rx = int(value.split()[0])
            return doc.xref_object(rx, compressed=True)
        except (ValueError, IndexError):
            return None
    return None


def _referenced_xrefs(dict_text):
    if not dict_text:
        return []
    return sorted({int(m.group(1)) for m in re.finditer(r"(\d+)\s+0\s+R", dict_text)})


def _is_image(doc, xref):
    return doc.xref_get_key(xref, "Subtype") == ("name", "/Image")


def _smask_stats(doc, xref):
    """Return (max_alpha, mean_alpha, width, height) for an image's SMask."""
    kind, value = doc.xref_get_key(xref, "SMask")
    if kind != "xref":
        return None
    try:
        smask_xref = int(value.split()[0])
    except (ValueError, IndexError):
        return None

    width = _xref_int(doc, smask_xref, "Width")
    height = _xref_int(doc, smask_xref, "Height")
    bpc = _xref_int(doc, smask_xref, "BitsPerComponent", 8)
    if not width or not height:
        return None

    try:
        data = doc.xref_stream(smask_xref)
    except Exception:
        data = b""

    if bpc == 8 and len(data) >= width * height:
        samples = data[: width * height]
    else:
        try:
            info = doc.extract_image(smask_xref)
            pix = pymupdf.Pixmap(info["image"])
            if pix.n != 1:
                pix = pymupdf.Pixmap(pymupdf.csGRAY, pix)
            samples = pix.samples
            width, height = pix.width, pix.height
        except Exception:
            return None

    if not samples:
        return None
    return max(samples), sum(samples) / len(samples), width, height


def detect_watermark_patterns(doc, min_alpha=4, max_alpha=200, max_image_px=2048):
    """Find tiling patterns that paint a faint, semi-transparent tiled image."""
    candidates = []
    for xref in range(1, doc.xref_length()):
        try:
            if doc.xref_get_key(xref, "Type") != ("name", "/Pattern"):
                continue
            if doc.xref_get_key(xref, "PatternType") != ("int", "1"):
                continue

            res_text = _resources_text(doc, xref)
            image_refs = [r for r in _referenced_xrefs(res_text) if _is_image(doc, r)]
            if not image_refs:
                continue

            for image_xref in image_refs:
                width = _xref_int(doc, image_xref, "Width", 0)
                height = _xref_int(doc, image_xref, "Height", 0)
                if width > max_image_px or height > max_image_px:
                    continue

                stats = _smask_stats(doc, image_xref)
                if not stats:
                    continue
                smask_max, smask_mean, _, _ = stats
                if not (min_alpha <= smask_max <= max_alpha):
                    continue

                candidates.append(
                    Candidate(
                        pattern_xref=xref,
                        image_xref=image_xref,
                        image_width=width,
                        image_height=height,
                        smask_max=smask_max,
                        smask_mean=smask_mean,
                        image_refs=image_refs,
                    )
                )
                break
        except Exception:
            continue
    return candidates


def clear_pattern(doc, xref):
    """Blank a tiling pattern and unlink its resources so the tile is purged."""
    doc.update_stream(xref, b"q\nQ\n")
    doc.xref_set_key(xref, "Resources", "<<>>")


@dataclass
class BakedCandidate:
    page_index: int
    image_xref: int
    width: int
    height: int
    basis: tuple
    mask_pixels: int
    active_phases: int
    repeats: float
    autocorr: float
    analysis: dict = field(default_factory=dict)
    rgb: object = None


def _dilate_bool(mask, radius):
    out = mask.copy()
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx or dy:
                out |= np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
    return out


def _box_sum(img, k):
    pad = k // 2
    p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    ii = np.pad(p, ((1, 0), (1, 0), (0, 0)), mode="constant").cumsum(0).cumsum(1)
    return ii[k:, k:, :] - ii[:-k, k:, :] - ii[k:, :-k, :] + ii[:-k, :-k, :]


def _local_max_2d(region, k=11):
    pad = k // 2
    p = np.pad(region, pad, mode="constant", constant_values=-np.inf)
    out = np.full(region.shape, -np.inf, dtype=region.dtype)
    for dy in range(k):
        for dx in range(k):
            out = np.maximum(out, p[dy : dy + region.shape[0], dx : dx + region.shape[1]])
    return out


def _faint_mask(rgb, gray):
    mx = rgb.max(axis=2).astype(np.int16)
    mn = rgb.min(axis=2).astype(np.int16)
    chroma = mx - mn
    mask = (gray >= 234) & (gray <= 251) & (chroma <= 5)
    dark = gray < 200
    mask &= ~_dilate_bool(dark, 2)
    rowsum = mask.sum(axis=1)
    colsum = mask.sum(axis=0)
    mask[rowsum > rgb.shape[1] * 0.15, :] = False
    mask[:, colsum > rgb.shape[0] * 0.15] = False
    return mask


def _periodic_basis(mask, max_len=900, min_score=0.12):
    h, w = mask.shape
    m = mask.astype(np.float64)
    spec = np.fft.rfft2(m)
    ac = np.fft.irfft2(spec * np.conj(spec), s=m.shape)
    ac = np.fft.fftshift(ac)
    cy, cx = h // 2, w // 2
    center = ac[cy, cx]
    if center <= 0:
        return None
    ac = ac / center

    y0 = max(0, cy - 400)
    y1 = min(h, cy + 401)
    x0 = max(0, cx - max_len)
    x1 = min(w, cx + max_len + 1)
    region = ac[y0:y1, x0:x1]
    local = _local_max_2d(region, 11)
    ys, xs = np.where((region >= local - 1e-9) & (region > min_score))
    if not len(ys):
        return None

    peaks = sorted(
        ((float(region[y, x]), int(x + x0 - cx), int(y + y0 - cy)) for y, x in zip(ys, xs)),
        reverse=True,
    )
    candidates = []
    for score, dx, dy in peaks:
        if abs(dx) <= 10 and abs(dy) <= 10:
            continue
        if any(abs(dx - ax) <= 3 and abs(dy - ay) <= 3 for _, ax, ay in candidates):
            continue
        candidates.append((score, dx, dy))
        if len(candidates) >= 12:
            break

    top = candidates[0][0] if candidates else 0.0
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            s1, x1v, y1v = candidates[i]
            s2, x2v, y2v = candidates[j]
            if s2 < top * 0.35:
                continue
            if abs(x1v * y2v - y1v * x2v) < 5000:
                continue
            return (x1v, y1v), (x2v, y2v), top
    return None


def _phase_indices(xs, ys, v1, v2):
    v1x, v1y = v1
    v2x, v2y = v2
    det = v1x * v2y - v1y * v2x
    if det == 0:
        return None
    u = (xs * v2y - ys * v2x) / det
    v = (-xs * v1y + ys * v1x) / det
    uf = u - np.floor(u)
    vf = v - np.floor(v)
    bu = int(abs(v1x) + abs(v2x)) + 2
    bv = int(abs(v1y) + abs(v2y)) + 2
    xi = np.clip((uf * bu).astype(np.int32), 0, bu - 1)
    yi = np.clip((vf * bv).astype(np.int32), 0, bv - 1)
    return yi, xi, bu, bv


def _analyze_baked(rgb):
    gray = (
        0.299 * rgb[:, :, 0].astype(np.float32)
        + 0.587 * rgb[:, :, 1].astype(np.float32)
        + 0.114 * rgb[:, :, 2].astype(np.float32)
    )
    gray_i = np.clip(gray, 0, 255).astype(np.int16)
    mask = _faint_mask(rgb, gray_i)
    if mask.sum() < max(500, rgb.shape[0] * rgb.shape[1] * 0.002):
        return None

    basis = _periodic_basis(mask)
    if not basis:
        return None
    v1, v2, top_score = basis
    if top_score < 0.2:
        return None

    ys, xs = np.nonzero(mask)
    mapped = _phase_indices(xs.astype(np.float64), ys.astype(np.float64), v1, v2)
    if not mapped:
        return None
    yi, xi, bu, bv = mapped
    count = np.zeros((bv, bu), np.int32)
    wmax = np.zeros((bv, bu), np.float32)
    np.add.at(count, (yi, xi), 1)
    np.maximum.at(wmax, (yi, xi), gray[ys, xs])

    active = count > 0
    active_phases = int(active.sum())
    if active_phases < 100:
        return None
    repeats = float(mask.sum()) / active_phases
    if repeats < 3.0:
        return None

    return {
        "gray": gray,
        "mask": mask,
        "v1": v1,
        "v2": v2,
        "count": count,
        "wmax": wmax,
        "bu": bu,
        "bv": bv,
        "top_score": top_score,
    }


def _render_baked_clean(rgb, analysis, alpha):
    gray = analysis["gray"]
    count = analysis["count"]
    wmax = analysis["wmax"]
    v1, v2 = analysis["v1"], analysis["v2"]
    h, w, _ = rgb.shape

    X = np.tile(np.arange(w, dtype=np.float64), h)
    Y = np.repeat(np.arange(h, dtype=np.float64), w)
    mapped = _phase_indices(X, Y, v1, v2)
    if not mapped:
        return None
    yi, xi, _, _ = mapped
    act = (count[yi, xi] > 0) & (wmax[yi, xi] < 252)
    if not act.any():
        return None
    W = wmax[yi, xi].reshape(h, w)

    bright = gray >= 250
    B0 = np.median(rgb.reshape(-1, 3)[bright.ravel()].astype(np.float32), axis=0)
    if not np.isfinite(B0).all():
        B0 = np.array([255.0, 255.0, 255.0], dtype=np.float32)

    rgb_f = rgb.astype(np.float32)
    sums = _box_sum(rgb_f * bright[..., None], 7)
    cnts = _box_sum(bright.astype(np.float32)[..., None], 7)
    bg = np.where(cnts >= 5, sums / np.maximum(cnts, 1.0), B0[None, None, :])

    corrected = B0[None, None, :] + (rgb_f - W[..., None]) / (1.0 - alpha)
    corrected = np.where(bright[..., None], bg, corrected)
    out = rgb_f.copy()
    out[act.reshape(h, w)] = corrected[act.reshape(h, w)]
    out = np.clip(out, 0, 255).astype(np.uint8)

    cleaned_gray = (
        0.299 * out[:, :, 0].astype(np.float32)
        + 0.587 * out[:, :, 1].astype(np.float32)
        + 0.114 * out[:, :, 2].astype(np.float32)
    )
    mask = analysis["mask"]
    residual = int(
        ((cleaned_gray >= 234) & (cleaned_gray <= 251))[mask].sum()
    )
    if residual > mask.sum() * 0.2:
        return None
    return out


def _load_rgb_image(doc, xref):
    if doc.xref_get_key(xref, "Subtype") != ("name", "/Image"):
        return None
    if doc.xref_get_key(xref, "SMask")[0] != "null":
        return None
    if doc.xref_get_key(xref, "Mask")[0] != "null":
        return None
    if doc.xref_get_key(xref, "ImageMask")[0] != "null":
        return None
    if doc.xref_get_key(xref, "Decode")[0] != "null":
        return None
    if _xref_int(doc, xref, "BitsPerComponent", 8) != 8:
        return None
    kind, cs = doc.xref_get_key(xref, "ColorSpace")
    if kind == "null" or not ("ICCBased" in cs or "DeviceRGB" in cs):
        return None
    try:
        pix = pymupdf.Pixmap(doc, xref)
    except Exception:
        return None
    if pix.n != 3 or pix.alpha:
        return None
    data = np.frombuffer(pix.samples, dtype=np.uint8)
    if pix.stride == pix.width * 3:
        arr = data.reshape(pix.height, pix.width, 3)
    else:
        arr = data.reshape(pix.height, pix.stride)[:, : pix.width * 3].reshape(
            pix.height, pix.width, 3
        )
    return arr.copy()


def find_baked_candidates(doc, max_pixels):
    found = []
    seen = set()
    for page_index in range(doc.page_count):
        page = doc[page_index]
        for item in page.get_images(full=True):
            xref = item[0]
            if xref in seen:
                continue
            seen.add(xref)
            if item[1]:
                continue
            if item[8] not in ("FlateDecode", ""):
                continue
            width, height = item[2], item[3]
            if width * height < 20000 or width * height > max_pixels:
                continue
            rgb = _load_rgb_image(doc, xref)
            if rgb is None:
                continue
            analysis = _analyze_baked(rgb)
            if not analysis:
                continue
            found.append(
                BakedCandidate(
                    page_index=page_index,
                    image_xref=xref,
                    width=width,
                    height=height,
                    basis=(analysis["v1"], analysis["v2"]),
                    mask_pixels=int(analysis["mask"].sum()),
                    active_phases=int((analysis["count"] > 0).sum()),
                    repeats=float(analysis["mask"].sum() / max(1, (analysis["count"] > 0).sum())),
                    autocorr=analysis["top_score"],
                    analysis=analysis,
                    rgb=rgb,
                )
            )
    return found


def describe_baked(candidates):
    print(f"[检测] 发现 {len(candidates)} 张内嵌图片含水印:")
    for c in candidates:
        v1, v2 = c.basis
        print(
            f"  - 第 {c.page_index + 1} 页 图片 xref {c.image_xref} "
            f"({c.width}x{c.height}px), 周期 {v1}/{v2}, "
            f"水印像素 {c.mask_pixels}, 重复约 {c.repeats:.1f} 次, "
            f"自相关 {c.autocorr:.2f}"
        )


def manual_candidates(doc, pattern_xrefs):
    candidates = []
    for xref in pattern_xrefs:
        res_text = _resources_text(doc, xref)
        refs = [r for r in _referenced_xrefs(res_text) if _is_image(doc, r)]
        if not refs:
            print(f"  [跳过] xref {xref}: 不是平铺图案或未引用图片")
            continue
        image_xref = refs[0]
        stats = _smask_stats(doc, image_xref)
        candidates.append(
            Candidate(
                pattern_xref=xref,
                image_xref=image_xref,
                image_width=_xref_int(doc, image_xref, "Width", 0),
                image_height=_xref_int(doc, image_xref, "Height", 0),
                smask_max=stats[0] if stats else -1,
                smask_mean=stats[1] if stats else -1.0,
                image_refs=refs,
            )
        )
    return candidates


def describe(candidates):
    print(f"[检测] 发现 {len(candidates)} 个疑似水印平铺图案:")
    for c in candidates:
        print(
            f"  - pattern xref {c.pattern_xref}: 贴图 xref {c.image_xref} "
            f"({c.image_width}x{c.image_height}px), alpha max={c.smask_max}/255, "
            f"mean={c.smask_mean:.2f}"
        )


def same_text(before, after):
    if len(before) != len(after):
        return False, f"页数不一致: {len(before)} -> {len(after)}"
    for i, (a, b) in enumerate(zip(before, after), 1):
        if a != b:
            return False, f"第 {i} 页文本不一致"
    return True, "逐页文本完全一致"


def main():
    parser = argparse.ArgumentParser(
        description="清除 PDF 中由平铺贴图实现的半透明水印（无损，保留文字矢量）"
    )
    parser.add_argument("input", help="输入 PDF 文件")
    parser.add_argument("-o", "--output", help="输出 PDF 文件（默认: 输入名 + -无水印.pdf）")
    parser.add_argument("--dry-run", action="store_true", help="只检测并列出候选对象，不写文件")
    parser.add_argument("--min-alpha", type=int, default=4, help="SMask 最大透明度下限（默认 4）")
    parser.add_argument("--max-alpha", type=int, default=200, help="SMask 最大透明度上限（默认 200）")
    parser.add_argument("--max-image-px", type=int, default=2048, help="贴图像素尺寸上限（默认 2048）")
    parser.add_argument("--pattern", help="手动指定 pattern xref 列表，如 82,83（跳过自动检测）")
    parser.add_argument(
        "--clean-baked",
        action="store_true",
        help="同时清除内嵌截图里“烤进去”的周期性文字水印（实验性，默认关闭）",
    )
    parser.add_argument(
        "--baked-alpha",
        type=float,
        default=0.1,
        help="内嵌图片水印的估计算子透明度（默认 0.1）",
    )
    parser.add_argument(
        "--baked-max-px",
        type=int,
        default=4000000,
        help="内嵌图片参与水印检测的最大像素数（默认 4,000,000）",
    )
    parser.add_argument("--force", action="store_true", help="输出文件已存在时覆盖")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"[错误] 找不到文件: {args.input}", file=sys.stderr)
        return 2

    output = args.output
    if not output:
        stem, _ = os.path.splitext(args.input)
        output = stem + "-无水印.pdf"

    if os.path.abspath(output) == os.path.abspath(args.input):
        print("[错误] 输出路径不能与输入相同，原文件不会被修改。", file=sys.stderr)
        return 2

    doc = pymupdf.open(args.input)
    if doc.needs_pass:
        print("[错误] PDF 已加密，请先解密。", file=sys.stderr)
        return 2

    try:
        if args.pattern:
            try:
                pattern_xrefs = [int(x) for x in args.pattern.replace("，", ",").split(",") if x.strip()]
            except ValueError:
                print("[错误] --pattern 需要逗号分隔的整数 xref。", file=sys.stderr)
                return 2
            candidates = manual_candidates(doc, pattern_xrefs)
        else:
            candidates = detect_watermark_patterns(
                doc, args.min_alpha, args.max_alpha, args.max_image_px
            )

        baked = []
        if args.clean_baked:
            baked = find_baked_candidates(doc, args.baked_max_px)

        if not candidates and not baked:
            print("[结果] 未检测到符合特征的水印，未生成输出。")
            print("       如确认存在平铺水印，可用 --pattern 指定对象 xref（先用 --dry-run 排查）。")
            return 1

        if candidates:
            describe(candidates)
        if baked:
            describe_baked(baked)
        if args.dry_run:
            print("[干跑] 未写入任何文件。")
            return 0
        if os.path.exists(output) and not args.force:
            print(f"[错误] 输出文件已存在: {output}（加 --force 覆盖）", file=sys.stderr)
            return 2

        before_texts = [page.get_text("text") for page in doc]

        for c in candidates:
            clear_pattern(doc, c.pattern_xref)
        if candidates:
            print(f"[处理] 已清空 {len(candidates)} 个水印图案并解除贴图引用")

        cleaned_count = 0
        for c in baked:
            cleaned = _render_baked_clean(c.rgb, c.analysis, args.baked_alpha)
            if cleaned is None:
                print(f"  [跳过] 图片 xref {c.image_xref}: 重建校验未通过")
                continue
            if doc.xref_get_key(c.image_xref, "DecodeParms")[0] != "null":
                doc.xref_set_key(c.image_xref, "DecodeParms", "null")
            doc.update_stream(c.image_xref, cleaned.tobytes(), compress=True)
            cleaned_count += 1
        print(f"[处理] 已重建 {cleaned_count} 张内嵌截图并写回 PDF")

        save_kwargs = dict(garbage=4, deflate=True)
        try:
            doc.save(output, use_objstms=1, **save_kwargs)
        except TypeError:
            doc.save(output, **save_kwargs)
    finally:
        doc.close()

    audit = pymupdf.open(output)
    try:
        after_texts = [page.get_text("text") for page in audit]
        ok, message = same_text(before_texts, after_texts)
        print(f"[校验] {message}" + ("" if ok else "（请人工复查）"))

        leftovers = detect_watermark_patterns(
            audit, args.min_alpha, args.max_alpha, args.max_image_px
        )
        if leftovers:
            print(f"[校验] 输出中仍有 {len(leftovers)} 个候选，请复查。")
        else:
            print("[校验] 输出中无残留候选水印图案。")

        if args.clean_baked:
            baked_left = find_baked_candidates(audit, args.baked_max_px)
            if baked_left:
                print(f"[校验] 输出中仍有 {len(baked_left)} 张图片检测到周期性水印，请复查。")
            else:
                print("[校验] 输出中无残留内嵌截图水印。")

        print(
            f"[完成] {output}  "
            f"({os.path.getsize(args.input) / 1024:.0f} KB -> {os.path.getsize(output) / 1024:.0f} KB)"
        )
    finally:
        audit.close()

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
