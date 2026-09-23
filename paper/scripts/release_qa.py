"""Small self-contained rendered-canvas check; no external skill dependency."""
def audit_layout(fig):
    from matplotlib.text import Text
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    # Tick artists beyond the displayed limits are retained but not drawn.
    undrawn_tick_labels = set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            lower, upper = sorted(axis.get_view_interval())
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lower <= tick.get_loc() <= upper:
                    undrawn_tick_labels.update((id(tick.label1), id(tick.label2)))
    for item in fig.findobj(Text):
        if id(item) in undrawn_tick_labels or not item.get_visible() or not item.get_text().strip():
            continue
        box = item.get_window_extent(renderer)
        if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.x1+1 or box.y1 > fig.bbox.y1+1):
            issues.append(("WARN", "Text outside canvas: " + item.get_text()))
    return issues
