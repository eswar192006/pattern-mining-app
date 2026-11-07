"""The charts: a plain ranking, plus two interactive 3D views of the same results.

  ranking_figure        horizontal bars, the quickest thing to read
  skyline_figure        a 3D "city skyline": one tower per combination, taller = stronger. Drag to rotate.
  constellation_figure  a 3D "star map": products are stars, combinations are the links between them, so
                        families of products that sell together form visible clusters. Drag, zoom, hover,
                        or press Spin.

Pure functions: result dict (from the API) in, Plotly figure out. No Streamlit, so they are tested directly.
Only plotly.graph_objects features that exist in Plotly 5, 6 and 7 are used.
"""
import math

import numpy as np
import plotly.graph_objects as go

from ui.present import combo_lines, combo_name, format_primary, kind_info, plain_metrics, primary_value, short

INK = "#c9d1f5"
GRID = "rgba(255,255,255,0.10)"
PALETTE = ["#7c5cff", "#22d3ee", "#ff5c9d", "#ffb454", "#4ade80", "#60a5fa", "#f472b6", "#a3e635"]
TOWER_SCALE = [[0.0, "#3b2bb3"], [0.45, "#7c5cff"], [0.75, "#c04bff"], [1.0, "#22d3ee"]]
FONT = dict(family="Inter, Segoe UI, Arial, sans-serif", color=INK)


def _rows(result: dict, top_n: int) -> list[dict]:
    return (result or {}).get("itemsets", [])[:top_n]


def _blank(scene: bool = True) -> dict:
    axis = dict(visible=False, showbackground=False, showgrid=False, zeroline=False)
    layout = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=FONT,
                  margin=dict(l=0, r=0, t=0, b=0), showlegend=False)
    if scene:
        layout["scene"] = dict(xaxis=axis, yaxis=dict(axis), zaxis=dict(axis), bgcolor="rgba(0,0,0,0)")
    return layout


# ---------------------------------------------------------------------------------------------- 2D ranking
def ranking_figure(result: dict, top_n: int = 10):
    """Horizontal bars, best at the top. None when there is nothing to show."""
    rows = _rows(result, top_n)
    if not rows:
        return None
    kind = result["kind"]
    values = [primary_value(kind, r) for r in rows]
    fig = go.Figure(go.Bar(
        x=values[::-1], y=[f"<b>#{r['rank']}</b>  {combo_lines(r)}" for r in rows][::-1], orientation="h",
        marker=dict(color=values[::-1], colorscale=TOWER_SCALE, line=dict(width=0)),
        text=[format_primary(kind, v) for v in values][::-1], textposition="outside", cliponaxis=False,
        hovertext=[f"<b>{combo_name(r)}</b><br>{plain_metrics(kind, r)}" for r in rows][::-1], hoverinfo="text"))
    fig.update_layout(**_blank(scene=False), height=max(280, 62 * len(rows) + 30),
                      xaxis=dict(visible=False, range=[0, (max(values) or 1.0) * 1.18]),
                      yaxis=dict(tickfont=dict(size=12, color=INK), automargin=True))
    fig.update_layout(margin=dict(l=0, r=30, t=10, b=0), bargap=0.32)
    return fig


# ---------------------------------------------------------------------------------------------- 3D skyline
def _tower(cx: float, cy: float, h: float, w: float = 0.72):
    """Vertices and triangles of a box standing on the floor (no bottom face)."""
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - w / 2, cy + w / 2
    xs = [x0, x1, x1, x0, x0, x1, x1, x0]
    ys = [y0, y0, y1, y1, y0, y0, y1, y1]
    zs = [0, 0, 0, 0, h, h, h, h]
    tris = [(4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return xs, ys, zs, [t[0] for t in tris], [t[1] for t in tris], [t[2] for t in tris]


def skyline_figure(result: dict, top_n: int = 12, columns: int = 4):
    """One tower per combination; height = the number that ranks them. Rank 1 stands at the front."""
    rows = _rows(result, top_n)
    if not rows:
        return None
    kind = result["kind"]
    values = [primary_value(kind, r) for r in rows]
    top, low = max(values) or 1.0, min(values)                           # (a zero top would divide by zero)
    fig = go.Figure()
    n_rows = math.ceil(len(rows) / columns)
    pad = 0.75
    fig.add_trace(go.Mesh3d(                                              # a faint floor to stand the towers on
        x=[-pad, columns - 1 + pad, columns - 1 + pad, -pad], y=[-pad, -pad, n_rows - 1 + pad, n_rows - 1 + pad],
        z=[0, 0, 0, 0], i=[0, 0], j=[1, 2], k=[2, 3], color="#7c5cff", opacity=0.10, hoverinfo="skip"))
    lab_x, lab_y, lab_z, lab_t = [], [], [], []
    for idx, (row, value) in enumerate(zip(rows, values)):
        cx, cy = idx % columns, (n_rows - 1) - idx // columns          # rank 1 at the front-left
        h = 0.25 + 2.75 * value / top
        xs, ys, zs, i, j, k = _tower(cx, cy, h)
        tint = 1.0 if top == low else 0.3 + 0.7 * (value - low) / (top - low)       # colour: relative to the other towers
        fig.add_trace(go.Mesh3d(
            x=xs, y=ys, z=zs, i=i, j=j, k=k, intensity=[0, 0, 0, 0, tint, tint, tint, tint],
            cmin=0, cmax=1, colorscale=TOWER_SCALE, showscale=False, flatshading=True,
            lighting=dict(ambient=0.62, diffuse=0.8, specular=0.25, roughness=0.55),
            hovertext=[f"<b>#{row['rank']} {combo_name(row)}</b><br>{plain_metrics(kind, row)}"] * 8, hoverinfo="text"))
        lab_x.append(cx), lab_y.append(cy), lab_z.append(h + 0.22), lab_t.append(f"#{row['rank']}")
    fig.add_trace(go.Scatter3d(x=lab_x, y=lab_y, z=lab_z, mode="text", text=lab_t, textfont=dict(size=13, color="#ffffff"),
                               hoverinfo="skip"))
    fig.update_layout(**_blank(), height=470)
    fig.update_layout(scene=dict(
        **_blank()["scene"], aspectmode="manual", aspectratio=dict(x=1.5, y=max(0.8, 0.55 * n_rows + 0.2), z=0.9),
        camera=dict(eye=dict(x=-1.55, y=-1.95, z=1.05), center=dict(x=0, y=0, z=-0.05))))
    return fig


# ---------------------------------------------------------------------------------------------- 3D constellation
def force_layout_3d(n: int, edges: list[tuple[int, int, float]], seed: int = 7, iterations: int = 220) -> np.ndarray:
    """Fruchterman-Reingold in 3D: every pair repels, linked pairs attract (stronger links pull harder).
    Deterministic for a given seed. Returns an (n, 3) array scaled to fit a sphere of radius 1."""
    if n == 0:
        return np.zeros((0, 3))
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3))
    k = 1.0
    for step in range(iterations):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-9
        disp = ((k * k / dist ** 2)[:, :, None] * delta).sum(axis=1)                  # repulsion
        for u, v, w in edges:
            d = pos[u] - pos[v]
            length = np.linalg.norm(d) + 1e-9
            pull = (length / k) * (0.5 + w) * d / length
            disp[u] -= pull
            disp[v] += pull
        disp -= 0.30 * pos                                                           # gravity: families stay close together
        temperature = 0.35 * (1 - step / iterations) + 0.01
        length = np.linalg.norm(disp, axis=1, keepdims=True) + 1e-9
        pos += disp / length * np.minimum(length, temperature)
    pos -= pos.mean(axis=0)
    return pos / (np.abs(np.linalg.norm(pos, axis=1)).max() + 1e-9)


def _components(n: int, edges) -> list[int]:
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for u, v, _ in edges:
        parent[find(u)] = find(v)
    roots = {}
    return [roots.setdefault(find(i), len(roots)) for i in range(n)]


def build_graph(result: dict, top_n: int = 25):
    """Products from the top combinations as nodes; every pair inside a combination is a link whose
    strength is the best (normalised) score of any combination containing both."""
    rows = _rows(result, top_n)
    kind = result["kind"] if result else "frequent"
    values = [primary_value(kind, r) for r in rows]
    top = (max(values) if values else 1.0) or 1.0
    names, index = [], {}
    weight, importance, appears = {}, {}, {}
    for row, value in zip(rows, values):
        score = value / top
        for p in row["itemset"]:
            if p not in index:
                index[p] = len(names)
                names.append(p)
            importance[p] = importance.get(p, 0.0) + score
            appears[p] = appears.get(p, 0) + 1
        items = row["itemset"]
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                key = tuple(sorted((index[items[a]], index[items[b]])))
                weight[key] = max(weight.get(key, 0.0), score)
    edges = [(u, v, w) for (u, v), w in weight.items()]
    return names, edges, importance, appears


def constellation_figure(result: dict, top_n: int = 25, spin: bool = True):
    """Products as stars, 'bought together' as links. Families of products form clusters (one colour each)."""
    rows = _rows(result, top_n)
    if not rows:
        return None
    kind = result["kind"]
    names, edges, importance, appears = build_graph(result, top_n)
    pos = force_layout_3d(len(names), edges)
    group = _components(len(names), edges)
    top_importance = max(importance.values()) or 1.0

    fig = go.Figure()
    tiers = [(0.0, 0.45, 2.5, 0.35), (0.45, 0.75, 4.5, 0.55), (0.75, 1.01, 7.0, 0.85)]       # (from, to, width, opacity)
    for lo, hi, width, opacity in tiers:
        xs, ys, zs = [], [], []
        for u, v, w in edges:
            if lo <= w < hi:
                xs += [pos[u][0], pos[v][0], None]
                ys += [pos[u][1], pos[v][1], None]
                zs += [pos[u][2], pos[v][2], None]
        if xs:
            fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", hoverinfo="skip",
                                       line=dict(width=width, color=f"rgba(201,209,245,{opacity})")))
    order = sorted(range(len(names)), key=lambda i: -importance[names[i]])
    labelled = set(order[:7])
    sizes = [13 + 22 * importance[n] / top_importance for n in names]
    colors = [PALETTE[g % len(PALETTE)] for g in group]
    fig.add_trace(go.Scatter3d(x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers", hoverinfo="skip",       # soft glow
                               marker=dict(size=[s_ * 2.1 for s_ in sizes], color=colors, opacity=0.13)))
    fig.add_trace(go.Scatter3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2], mode="markers+text",
        text=[short(names[i], 18) if i in labelled else "" for i in range(len(names))],
        textposition=["top center" if i % 2 == 0 else "bottom center" for i in range(len(names))],
        textfont=dict(size=11, color="#e6eaff"),
        marker=dict(size=sizes, color=colors, opacity=0.97, line=dict(width=1.5, color="rgba(255,255,255,0.7)")),
        hovertext=[f"<b>{n}</b><br>in {appears[n]} of the top {len(rows)} combinations" for n in names], hoverinfo="text"))
    fig.update_layout(**_blank(), height=520)
    r = 1.4
    fig.update_layout(scene=dict(**_blank()["scene"], aspectmode="cube",
                                 camera=dict(eye=dict(x=r, y=0.6, z=0.7))))
    if spin:
        frames = []
        for deg in range(0, 360, 6):
            a = math.radians(deg)
            frames.append(go.Frame(layout=dict(scene=dict(camera=dict(eye=dict(x=r * math.cos(a), y=r * math.sin(a), z=0.7))))))
        fig.frames = frames
        play = dict(frame=dict(duration=70, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")
        stop = dict(frame=dict(duration=0, redraw=False), mode="immediate", transition=dict(duration=0))
        fig.update_layout(updatemenus=[dict(
            type="buttons", direction="left", showactive=False, x=0.01, y=0.02, xanchor="left", yanchor="bottom",
            bgcolor="rgba(124,92,255,0.25)", bordercolor="rgba(124,92,255,0.8)", font=dict(color="#ffffff", size=12),
            buttons=[dict(label="▶  Spin", method="animate", args=[None, play]),
                     dict(label="⏸  Pause", method="animate", args=[[None], stop])])])
    return fig
