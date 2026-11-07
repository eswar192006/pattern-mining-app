"""The charts (ui/charts.py): structure of each figure, and the edge cases that once looked odd.

No browser is needed: a Plotly figure is plain data, so we check the traces, sizes and labels directly.
Cases: empty results, one result, triples, and all three kinds of pattern.
"""
import numpy as np
import pytest

pytest.importorskip("plotly")

from tests.ui_samples import result  # noqa: E402
from ui import charts  # noqa: E402

KINDS = ("frequent", "rare", "high_utility")


def products_in(res):
    return {p for r in res["itemsets"] for p in r["itemset"]}


# ------------------------------------------------------------------ empty and degenerate input
def test_every_chart_returns_none_when_there_is_nothing_to_show():
    for res in (result(n=0), {"kind": "frequent", "itemsets": []}, {}, None):
        assert charts.ranking_figure(res) is None
        assert charts.skyline_figure(res) is None
        assert charts.constellation_figure(res) is None


def test_zero_values_do_not_divide_by_zero():
    res = result("frequent", 2)
    for r in res["itemsets"]:
        r["support"] = 0.0
    for build in (charts.ranking_figure, charts.skyline_figure, charts.constellation_figure):
        assert build(res) is not None


# ------------------------------------------------------------------ ranking
def test_ranking_has_one_bar_per_combination_with_the_best_on_top():
    for kind in KINDS:
        res = result(kind, 6)
        fig = charts.ranking_figure(res)
        bar = fig.data[0]
        assert bar.type == "bar" and bar.orientation == "h" and len(bar.x) == 6
        assert bar.x[-1] == max(bar.x)                       # plotly draws the last bar at the top: rank 1
        assert "#1" in bar.y[-1] and "#6" in bar.y[0]


def test_ranking_caps_at_top_n_and_handles_a_single_result():
    assert len(charts.ranking_figure(result("frequent", 15), top_n=10).data[0].x) == 10
    assert len(charts.ranking_figure(result("frequent", 1)).data[0].x) == 1


def test_ranking_labels_keep_every_product_of_a_triple():
    fig = charts.ranking_figure(result("frequent", 3, size=3))
    assert all(label.count("<br>+") == 2 for label in fig.data[0].y)


# ------------------------------------------------------------------ skyline
def test_skyline_has_a_floor_one_tower_per_combination_and_labels():
    for kind in KINDS:
        fig = charts.skyline_figure(result(kind, 7))
        types = [t.type for t in fig.data]
        assert types == ["mesh3d"] + ["mesh3d"] * 7 + ["scatter3d"]
        assert all(len(t.x) == 8 for t in fig.data[1:8])     # each tower is a box with 8 corners
        assert list(fig.data[-1].text) == [f"#{i}" for i in range(1, 8)]


def test_skyline_towers_get_shorter_with_rank():
    fig = charts.skyline_figure(result("frequent", 6))
    heights = [max(t.z) for t in fig.data[1:7]]
    assert heights == sorted(heights, reverse=True) and len(set(heights)) == 6


def test_skyline_single_result_and_triples_and_hover_text():
    one = charts.skyline_figure(result("high_utility", 1))
    assert len(one.data) == 3 and max(one.data[1].z) > 2.9                       # a lone tower is full height
    triple = charts.skyline_figure(result("rare", 2, size=3))
    assert triple.data[1].hovertext[0].count(" + ") == 2


def test_skyline_caps_the_number_of_towers():
    assert len(charts.skyline_figure(result("frequent", 30), top_n=12).data) == 14


# ------------------------------------------------------------------ star map
def test_star_map_has_one_star_per_product_and_a_spin_button():
    for kind in KINDS:
        res = result(kind, 6)
        fig = charts.constellation_figure(res)
        stars = [t for t in fig.data if t.type == "scatter3d" and t.mode == "markers+text"][0]
        assert len(stars.x) == len(products_in(res))
        assert len(fig.frames) == 60
        labels = [b.label for b in fig.layout.updatemenus[0].buttons]
        assert any("Spin" in l for l in labels) and any("Pause" in l for l in labels)


def test_star_map_without_spin_has_no_animation():
    fig = charts.constellation_figure(result("frequent", 4), spin=False)
    assert len(fig.frames) == 0 and not fig.layout.updatemenus


def test_star_map_single_result_and_triples():
    one = charts.constellation_figure(result("frequent", 1))
    assert [t for t in one.data if t.mode == "markers+text"][0].x.shape == (2,)
    names, edges, importance, appears = charts.build_graph(result("frequent", 1, size=3))
    assert len(names) == 3 and len(edges) == 3                  # a triple links all three pairs


def test_graph_weights_are_relative_to_the_best_combination():
    res = result("frequent", 4)
    names, edges, importance, appears = charts.build_graph(res)
    assert max(w for _, _, w in edges) == 1.0 and all(0 < w <= 1.0 for _, _, w in edges)
    assert set(names) == products_in(res) and all(appears[n] >= 1 for n in names)


# ------------------------------------------------------------------ the layout maths
def test_layout_is_deterministic_and_fits_a_unit_sphere():
    edges = [(0, 1, 1.0), (1, 2, 0.5), (3, 4, 0.8)]
    a, b = charts.force_layout_3d(5, edges), charts.force_layout_3d(5, edges)
    assert a.shape == (5, 3) and np.allclose(a, b)
    assert np.linalg.norm(a, axis=1).max() <= 1.0 + 1e-9
    assert charts.force_layout_3d(0, []).shape == (0, 3) and charts.force_layout_3d(1, []).shape == (1, 3)


def test_linked_products_end_up_closer_than_unlinked_ones():
    edges = [(0, 1, 1.0), (1, 2, 1.0), (0, 2, 1.0), (3, 4, 1.0), (4, 5, 1.0), (3, 5, 1.0)]
    pos = charts.force_layout_3d(6, edges)
    inside = np.linalg.norm(pos[0] - pos[1])
    between = np.linalg.norm(pos[0] - pos[3])
    assert inside < between


def test_product_families_get_their_own_colour_group():
    groups = charts._components(5, [(0, 1, 1.0), (1, 2, 1.0), (3, 4, 1.0)])
    assert groups[0] == groups[1] == groups[2] and groups[3] == groups[4] and groups[0] != groups[3]
