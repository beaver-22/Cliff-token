"""Renderer for ``fig4_self_correction`` and its companion ``fig4_self_correction_legend``.

The figure lays a whole model response out as flowing text -- line breaks
collapsed, markdown scaffolding dropped, ``$...$`` spans typeset as real math
and ``**...**`` spans set in bold -- and shades every token (or formula) by its
estimated success probability: red = low, green = high.  Marked tokens are
boxed: the statistical cliff tokens, the token where the wrong value is
written, and the self-correction cue.

Input is the self-contained snapshot written by
``scripts/build_self_correction_case.py`` to
``figure/data/fig4_self_correction/trace.json``, so the notebook that calls
``render`` reads nothing outside ``figure/``.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory

STEM = 'fig4_self_correction'
LEGEND_STEM = 'fig4_self_correction_legend'

FIG_W = 7.0                 # inches; wider than the 5.5in \textwidth, so place
                            # it 1:1 with \makebox[\textwidth][c]{\includegraphics}
FONT_SIZE = 7.0             # points, at the authored figure width
# Aptos ships with Microsoft 365 and is not redistributable, so it is used only
# when the machine running this script already has it (the ICLR notebooks pick
# it up the same way).  Otherwise: STIXGeneral = Times metrics, matching the
# paper body text; DejaVu Sans = what fig01-fig17 currently use.
FONT_PREFERENCE = ['Aptos', 'STIXGeneral']
INK = '#141414'             # a shade off pure black
ELIDE_GREY = '#ececec'      # the collapsed span, in both panels
MARGIN = 0.06               # side margin, in inches
ROW_PAD = 0.08              # blank band above/below the glyphs, in em
ROW_GAP = 0.50              # gap between two coloured bands, in em
MARK_GAP = 0.75             # extra gap under a line that carries a box, in em
ANNOT_GAP = 2.30            # extra gap under a line carrying a callout, in em
MATH_PAD_EM = 0.16          # breathing room around a formula, in em
SHOW_CALLOUTS = True        # draw the leader lines + labels under each boxed span
LEGEND_VERTICAL = True      # legend orientation: vertical bar, ticks on the right
SHOW_CURVE = True           # success-probability panel above the shaded text
SHOW_COLORBAR = True        # colour key inside the figure (legend file is still written)
CURVE_H = 1.15              # height of the probability panel, in inches
CURVE_GAP = 0.34            # gap between the panel and the text block, in inches
ELIDE_GAP = 18              # width of the collapsed span, in token units

ROLE_COLOR = {'cliff': '#ff2600', 'cue': '#1565c0', 'error': '#000000'}
CMAP = LinearSegmentedColormap.from_list(
    'pot', ['#ef8a84', '#f8b57f', '#fbe08c', '#d4e79a', '#8ed48d']
)
# markdown scaffolding; dropped only at the start of a source line, so that
# ordinary math such as "$ b > 9 $" survives untouched
LINE_START_MARKERS = {'---', '###', '##', '#', '>'}
# rewrites that let matplotlib's mathtext engine parse the model's LaTeX
MATH_FIXES = [(r'\\\\', ';'), (r'\\boxed\{([^{}]*)\}', r'\\mathbf{\1}')]
SENTINEL = '|'              # brackets a string so leading/trailing spaces count


def resolve_font() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in FONT_PREFERENCE if name in installed), 'DejaVu Sans')


def _flatten_frac(match: re.Match) -> str:
    """Set a fraction inline, so one formula never makes its row taller."""
    parts = []
    for part in (match.group(1), match.group(2)):
        part = part.strip()
        parts.append(part if re.fullmatch(r'[\w.]+', part) else f'({part})')
    return f'{parts[0]}/{parts[1]}'


def fix_math(expr: str) -> str:
    for pattern, repl in MATH_FIXES:
        expr = re.sub(pattern, repl, expr)
    while True:
        flattened = re.sub(r'\\frac\{([^{}]*)\}\{([^{}]*)\}', _flatten_frac, expr)
        if flattened == expr:
            return expr.strip()
        expr = flattened


def load_snapshot(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def _draw_curve(ax, scores, marks, elide, text_font, font_size) -> None:
    """Success probability against token position, above the shaded text.

    The span the text panel elides is collapsed here too, so a position on the
    curve lines up with the same moment in the response.  Every boxed span in
    the text gets a tick of the same colour, and the cliff / recovery pairs are
    annotated with the numbers quoted in the paper.
    """
    elide_after, elide_before = elide
    skipped = (elide_before - elide_after) - ELIDE_GAP

    def to_x(position: int) -> float:
        return position if position < elide_before else position - skipped

    kept = [p for p in range(1, len(scores) + 1)
            if p <= elide_after or p >= elide_before]
    ax.plot([to_x(p) for p in kept], [scores[p - 1] for p in kept],
            lw=0.7, color=INK, solid_joinstyle='round', zorder=3)
    # the collapsed span: same grey as the elision marker in the text panel,
    # with an axis break on the x spine so the jump in position is explicit
    band_a, band_b = to_x(elide_after), to_x(elide_before)
    ax.axvspan(band_a, band_b, color=ELIDE_GREY, lw=0, zorder=0)
    for edge in (band_a, band_b):
        ax.plot([edge, edge], [0, 1], lw=0.5, ls=(0, (1.2, 1.4)), color='#999999',
                zorder=1)
    mid = (band_a + band_b) / 2
    blended = blended_transform_factory(ax.transData, ax.transAxes)
    slant = 0.30 * (band_b - band_a)           # two slashes across the x spine
    for centre in (mid - 0.22 * (band_b - band_a), mid + 0.22 * (band_b - band_a)):
        ax.plot([centre - slant / 2, centre + slant / 2], [-0.055, 0.055],
                transform=blended, lw=0.7, color=INK, clip_on=False,
                solid_capstyle='round', zorder=6)
    ax.text(mid, 1.03, f'{elide_before - elide_after} tokens elided',
            fontsize=font_size * 0.8, color='#777777', ha='center', va='bottom',
            family=text_font, style='italic', parse_math=False, transform=blended)

    for mark in marks:
        start, end = mark['span']
        prev = scores[max(start - 2, 0)]
        curr = scores[min(end, len(scores)) - 1]
        x = to_x(end)
        ax.plot([x, x], [0, 1], lw=0.7, ls=(0, (2.2, 1.6)), color=mark['color'],
                alpha=0.75, zorder=2)
        ax.plot([x], [curr], marker='o', ms=2.6, color=mark['color'],
                mec='white', mew=0.4, zorder=4)
        if mark['role'] == 'error':
            continue
        falling = curr < prev
        ax.annotate(f'{prev:.2f}\u2192{curr:.2f}', xy=(x, curr),
                    xytext=(-3 if falling else 3, -9 if falling else 7),
                    textcoords='offset points', fontsize=font_size * 0.85,
                    color=mark['color'], ha='right' if falling else 'left',
                    va='top' if falling else 'bottom', family=text_font,
                    fontweight='bold', parse_math=False, zorder=5)

    ax.set_xlim(0, to_x(len(scores)))
    ax.set_ylim(-0.02, 1.02)
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_ylabel('success probability', fontsize=font_size, color=INK, labelpad=2)
    ax.set_xlabel('token position in the response (elided span collapsed)',
                  fontsize=font_size, color=INK, labelpad=1.2)
    ax.tick_params(labelsize=font_size * 0.85, length=2, pad=1.5, colors=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_linewidth(0.5)
        ax.spines[side].set_color(INK)


def render(snapshot: dict, out_dir: str | Path, fig_w: float = FIG_W,
           font_size: float = FONT_SIZE, show_callouts: bool = SHOW_CALLOUTS,
           labels: list[str] | None = None, show_curve: bool = SHOW_CURVE,
           show_colorbar: bool = SHOW_COLORBAR) -> dict:
    """Write ``fig4_self_correction`` and ``fig4_self_correction_legend`` (PNG + PDF) to ``out_dir``.

    ``labels`` overrides the per-mark callout text carried by the snapshot, so the
    notebook can keep the wording in step with the paper without rebuilding the
    snapshot.  Returns a small summary dict describing what was laid out.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokens: list[str] = snapshot['tokens']
    scores: list[float] = snapshot['scores']
    elide_after, elide_before = snapshot['elide']
    text_font = resolve_font()
    math_fontset = 'stix' if text_font == 'STIXGeneral' else 'dejavusans'

    def potential(position: int) -> float:
        return scores[min(position, len(scores)) - 1]

    marks = [dict(mark, color=ROLE_COLOR[mark['role']]) for mark in snapshot['marks']]
    for mark, label in zip(marks, labels or []):
        mark['label'], mark['value'] = label, ''
    marked = {p: m for m in marks for p in range(m['span'][0], m['span'][1] + 1)}

    # -------------------------------------------- flatten into a char stream
    elided = [s for s in scores[elide_after - 1:elide_before - 1] if s is not None]
    elide_text = f'  [ ... {elide_before - elide_after} tokens elided ... ]  '

    chars: list[str] = []
    owner: list[int] = []
    prev_space = True
    for position, token in enumerate(tokens, start=1):
        if elide_after <= position < elide_before:
            if position == elide_after:
                chars.extend(elide_text)
                owner.extend([-1] * len(elide_text))
                prev_space = True
            continue
        at_line_start = position == 1 or tokens[position - 2].endswith('\n')
        if at_line_start and token.strip() in LINE_START_MARKERS:
            continue
        shown = re.sub(r'\s+', ' ', token).replace('\ufffd', '')
        for ch in shown:
            if ch == ' ' and prev_space:
                continue
            chars.append(ch)
            owner.append(position)
            prev_space = ch == ' '

    text = ''.join(chars)
    math_span: list[int | None] = [None] * len(text)
    is_bold = [False] * len(text)
    drop = [False] * len(text)                 # delimiters, not shown

    for index, match in enumerate(re.finditer(r'\$\$(.+?)\$\$|\$(.+?)\$', text)):
        a, b = match.span()
        inner_a = a + (2 if match.group(1) is not None else 1)
        inner_b = b - (2 if match.group(1) is not None else 1)
        for k in range(a, inner_a):
            drop[k] = True
        for k in range(inner_b, b):
            drop[k] = True
        for k in range(inner_a, inner_b):
            math_span[k] = index
    for match in re.finditer(r'\*\*(.+?)\*\*', text):
        a, b = match.span()
        if any(math_span[k] is not None for k in (a, b - 1)):
            continue
        for k in (a, a + 1, b - 2, b - 1):
            drop[k] = True
        for k in range(a + 2, b - 2):
            is_bold[k] = True

    # ------------------------------------------------------------ build atoms
    # an atom is one drawing unit: a plain-text token, or a whole formula
    atoms: list[dict] = []
    k = 0
    while k < len(text):
        if drop[k]:
            k += 1
            continue
        if math_span[k] is not None:
            span_id = math_span[k]
            j = k
            while j < len(text) and math_span[j] == span_id:
                j += 1
            positions = sorted({owner[i] for i in range(k, j)})
            hits = [p for p in positions if p in marked]
            pieces = [(k, j, None)]
            if hits:                     # split so the marked token can be boxed
                cut_a = next(i for i in range(k, j) if owner[i] == hits[0])
                cut_b = next(i for i in range(j, k, -1) if owner[i - 1] == hits[-1])
                pieces = [(k, cut_a, None), (cut_a, cut_b, marked[hits[0]]), (cut_b, j, None)]
                pieces = [p for p in pieces if p[1] > p[0]]
            for a, b, mark in pieces:
                body = fix_math(text[a:b])
                if not body:
                    continue
                pots = [potential(owner[i]) for i in range(a, b)]
                atoms.append({'kind': 'math', 'body': body, 'raw': text[a:b],
                              'pot': sum(pots) / len(pots), 'mark': mark,
                              'pad_l': float(a == k), 'pad_r': float(b == j),
                              'group': span_id, 'breakable': a == k})
            k = j
            continue
        j = k
        while (j < len(text) and math_span[j] is None and not drop[j]
               and owner[j] == owner[k] and is_bold[j] == is_bold[k]):
            j += 1
        body = text[k:j]
        atoms.append({'kind': 'text', 'body': body, 'raw': body,
                      'pot': None if owner[k] < 0 else potential(owner[k]),
                      'mark': marked.get(owner[k]), 'bold': is_bold[k],
                      'italic': owner[k] < 0, 'group': None,
                      'breakable': body.startswith(' ')})
        k = j

    # ----------------------------------------------------------------- layout
    ax_w_frac = 1.0 - 2 * MARGIN / fig_w
    text_w = fig_w * ax_w_frac
    em = font_size / 72.0                      # inches
    math_pad = MATH_PAD_EM * em

    with matplotlib.rc_context({'mathtext.fontset': math_fontset}):
        probe = plt.figure(figsize=(fig_w, 4), dpi=300)
        probe.canvas.draw()
        renderer = probe.canvas.get_renderer()
        cache: dict[tuple, tuple[float, float, float]] = {}

        def _extent(body: str, **kwargs) -> tuple[float, float, float]:
            """(width, ascent, descent) of one string, in inches."""
            artist = probe.text(0.5, 0.5, body, fontsize=font_size, va='baseline', **kwargs)
            try:
                box = artist.get_window_extent(renderer)
                anchor = probe.transFigure.transform((0.5, 0.5))[1]
                return (box.width / probe.dpi, (box.y1 - anchor) / probe.dpi,
                        (anchor - box.y0) / probe.dpi)
            finally:
                artist.remove()

        def measure(atom: dict) -> tuple[float, float, float]:
            """Return (width, ascent, descent) of an atom, in inches."""
            key = (atom['kind'], atom['body'], atom.get('bold', False),
                   atom.get('italic', False))
            if key in cache:
                return cache[key]
            if atom['kind'] == 'text':
                style = dict(
                    family=text_font, parse_math=False,
                    fontweight='bold' if atom.get('bold') else 'normal',
                    fontstyle='italic' if atom.get('italic') else 'normal',
                )
                padded, ascent, descent = _extent(SENTINEL + atom['body'] + SENTINEL, **style)
                bare = _extent(SENTINEL + SENTINEL, **style)[0]
                result = (max(padded - bare, 0.0), ascent, descent)
            else:
                try:
                    result = _extent(f"${atom['body']}$")
                except Exception:              # mathtext cannot parse it
                    atom['kind'] = 'text'
                    atom['body'] = atom['raw']
                    return measure(atom)
            result = (result[0], max(result[1], 0.70 * em), max(result[2], 0.22 * em))
            cache[key] = result
            return result

        for atom in atoms:
            atom['w'], atom['asc'], atom['desc'] = measure(atom)
            if atom['kind'] == 'math':
                atom['pad_l'] = atom.get('pad_l', 1.0) * math_pad
                atom['pad_r'] = atom.get('pad_r', 1.0) * math_pad
                atom['w'] += atom['pad_l'] + atom['pad_r']

        def _chunks(items: list[dict]):
            """Yield atoms that must share a line (one formula = one chunk)."""
            index = 0
            while index < len(items):
                group = items[index]['group']
                start = index
                index += 1
                if group is not None:
                    while index < len(items) and items[index]['group'] == group:
                        index += 1
                yield items[start:index]

        rows: list[list[dict]] = [[]]
        x = 0.0
        for chunk in _chunks(atoms):
            width = sum(a['w'] for a in chunk)
            if (x + width > text_w and chunk[0]['breakable'] and rows[-1]
                    and width <= text_w):
                head = chunk[0]
                if head['kind'] == 'text':
                    head['body'] = head['body'].lstrip(' ')
                    head['w'] = measure(head)[0]
                rows.append([])
                x = 0.0
            for atom in chunk:
                atom['x'] = x
                rows[-1].append(atom)
                x += atom['w']
        plt.close(probe)

        y, baseline = 0.0, []
        for row in rows:
            asc = max(a['asc'] for a in row) + ROW_PAD * em
            desc = max(a['desc'] for a in row) + ROW_PAD * em
            y += asc
            baseline.append((y, asc, desc))
            y += desc + ROW_GAP * em
            if any(a['mark'] for a in row):
                y += (ANNOT_GAP if show_callouts else MARK_GAP) * em
        text_h = y

        top, bottom = 0.10, 0.34 if show_colorbar else 0.10
        curve_block = (CURVE_H + CURVE_GAP) if show_curve else 0.0
        fig_h = top + curve_block + text_h + bottom
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=300)
        norm = Normalize(0.0, 1.0)

        # ------------------------------------ success-probability panel (top)
        if show_curve:
            cax_curve = fig.add_axes([MARGIN / fig_w, (bottom + text_h + CURVE_GAP) / fig_h,
                                      ax_w_frac, CURVE_H / fig_h])
            _draw_curve(cax_curve, scores, marks, (elide_after, elide_before),
                        text_font, font_size)

        # ------------------------------------------------------- text panel
        ax = fig.add_axes([MARGIN / fig_w, bottom / fig_h, ax_w_frac, text_h / fig_h])
        ax.set_xlim(0, text_w)
        ax.set_ylim(text_h, 0)
        ax.axis('off')

        for row, (y_base, asc, desc) in zip(rows, baseline):
            for atom in row:
                ax.add_patch(Rectangle(            # the elision marker stays white
                    (atom['x'], y_base - asc), atom['w'], asc + desc,
                    facecolor=ELIDE_GREY if atom['pot'] is None else CMAP(norm(atom['pot'])),
                    edgecolor='none', zorder=1,
                ))
                if atom['kind'] == 'math':
                    ax.text(atom['x'] + atom['pad_l'], y_base, f"${atom['body']}$",
                            fontsize=font_size, color=INK, va='baseline', ha='left',
                            zorder=3)
                else:
                    ax.text(atom['x'], y_base, atom['body'], family=text_font,
                            fontsize=font_size, color=INK, va='baseline', ha='left',
                            zorder=3,
                            fontweight='bold' if atom.get('bold') else 'normal',
                            fontstyle='italic' if atom.get('italic') else 'normal',
                            parse_math=False)

            for mark in marks:                            # boxes and callouts
                member = [a for a in row if a['mark'] is mark]
                if not member:
                    continue
                x0 = min(a['x'] for a in member)
                x1 = max(a['x'] + a['w'] for a in member)
                ax.add_patch(Rectangle(
                    (x0 - 0.08 * em, y_base - asc + 0.04 * em), x1 - x0 + 0.16 * em,
                    asc + desc - 0.08 * em, facecolor='none', edgecolor=mark['color'],
                    lw=0.9, zorder=4,
                ))
                if not show_callouts:
                    continue
                label = f"{mark['label']}  {mark.get('value', '')}".rstrip()
                half = len(label) * 0.30 * em
                cx = min(max((x0 + x1) / 2, half), text_w - half)
                ax.plot([(x0 + x1) / 2] * 2,
                        [y_base + desc + 0.10 * em, y_base + desc + 0.80 * em],
                        lw=0.8, color=mark['color'], zorder=4)
                ax.text(cx, y_base + desc + 0.92 * em, label, fontsize=font_size * 1.10,
                        color=mark['color'], ha='center', va='top', zorder=5,
                        fontweight='bold', family=text_font, parse_math=False)

        if show_colorbar:
            # colour key for the text shading, right-aligned in the band between
            # the probability panel and the first line of text
            key_w, key_h = 1.10 / fig_w, 0.070 / fig_h
            key = fig.add_axes([1 - MARGIN / fig_w - key_w, 0.115 / fig_h,
                                key_w, key_h])
            inline = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=CMAP), cax=key,
                                  orientation='horizontal')
            inline.set_ticks([0.0, 1.0])
            inline.ax.tick_params(labelsize=font_size * 0.85, length=0, pad=1.0)
            inline.outline.set_linewidth(0.4)
            key.text(-0.04, 0.5, 'success probability',
                     transform=key.transAxes, fontsize=font_size * 0.9, color=INK,
                     ha='right', va='center')

        fig.savefig(out_dir / f'{STEM}.png', bbox_inches='tight', pad_inches=0.03)
        fig.savefig(out_dir / f'{STEM}.pdf', bbox_inches='tight', pad_inches=0.03)
        plt.close(fig)

        # ---------------------------------------- stand-alone legend
        if LEGEND_VERTICAL:
            legend = plt.figure(figsize=(0.62, 1.9), dpi=300)
            cax = legend.add_axes([0.06, 0.02, 0.32, 0.96])
        else:
            legend = plt.figure(figsize=(1.9, 0.44), dpi=300)   # = 0.35\textwidth
            cax = legend.add_axes([0.02, 0.46, 0.96, 0.50])
        bar = legend.colorbar(
            cm.ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
            orientation='vertical' if LEGEND_VERTICAL else 'horizontal',
        )
        bar.set_ticks([0, 0.5, 1.0])
        bar.ax.tick_params(labelsize=font_size, length=2, pad=1.5)
        bar.outline.set_linewidth(0.4)
        legend.savefig(out_dir / f'{LEGEND_STEM}.png', bbox_inches='tight', pad_inches=0.02)
        legend.savefig(out_dir / f'{LEGEND_STEM}.pdf', bbox_inches='tight', pad_inches=0.02)
        plt.close(legend)

    return {
        'font': text_font,
        'atoms': len(atoms),
        'formulas': sum(1 for a in atoms if a['kind'] == 'math'),
        'lines': len(rows),
        'figure_inches': (round(float(fig_w), 2), round(float(fig_h), 2)),
        'elided_tokens': len(elided),
        'elided_probability': (round(min(elided), 3), round(max(elided), 3)),
    }
