import numpy as np
import pytest

from flowdesk.core.gates import (
    BooleanGate,
    EllipseGate,
    PolygonGate,
    RectangleGate,
    gate_from_dict,
    make_quadrant_gates,
    points_in_polygon,
)
from flowdesk.core.transforms import LinearScale, LogScale


class FakeAccess:
    """Minimal ParamAccess: data columns, per-parameter scales, named masks."""

    def __init__(self, columns: dict[str, list[float]], scales: dict | None = None, masks: dict | None = None):
        self.columns = {k: np.asarray(v, dtype=float) for k, v in columns.items()}
        self.scales = scales or {}
        self.masks = masks or {}

    def _scale(self, name):
        return self.scales.get(name, LinearScale(0.0, 1.0))  # identity

    def data(self, name):
        if name not in self.columns:
            raise KeyError(name)
        return self.columns[name]

    def display(self, name):
        return self._scale(name).forward(self.data(name))

    def to_display(self, name, values):
        return self._scale(name).forward(np.asarray(values, dtype=float))

    def resolve(self, path):
        m = self.masks.get(path)
        return None if m is None else np.asarray(m, dtype=bool)

    @property
    def n_events(self):
        return len(next(iter(self.columns.values()))) if self.columns else 0


# ------------------------------------------------------------- point in polygon
def test_points_in_polygon_square():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    px = np.array([5, 0.1, 9.9, -1, 11, 5, 5])
    py = np.array([5, 0.1, 9.9, 5, 5, -0.1, 10.1])
    np.testing.assert_array_equal(points_in_polygon(px, py, sq), [True, True, True, False, False, False, False])


def test_points_in_polygon_concave():
    # L shape: the notch (7, 7) is outside
    L = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10)]
    px = np.array([2, 8, 2, 7, 3.9])
    py = np.array([8, 2, 2, 7, 9.9])
    np.testing.assert_array_equal(points_in_polygon(px, py, L), [True, True, True, False, True])


def test_points_in_polygon_triangle_and_degenerate():
    tri = [(0, 0), (4, 0), (0, 4)]
    np.testing.assert_array_equal(points_in_polygon(np.array([1.0, 3.0]), np.array([1.0, 3.0]), tri), [True, False])
    assert not points_in_polygon(np.array([0.5]), np.array([0.5]), [(0, 0), (1, 1)]).any()
    assert points_in_polygon(np.array([]), np.array([]), tri).shape == (0,)


# ------------------------------------------------------------------- rectangles
def test_rectangle_1d_half_open():
    acc = FakeAccess({"X": [0, 1, 2, 3, 4, 5]})
    g = RectangleGate(["X"], [1], [4])
    np.testing.assert_array_equal(g.evaluate(acc), [False, True, True, True, False, False])
    assert g.is_range and g.is_1d


def test_rectangle_open_bounds():
    acc = FakeAccess({"X": [-100, 0, 5, 1e9]})
    np.testing.assert_array_equal(RectangleGate(["X"], [None], [5]).evaluate(acc), [True, True, False, False])
    np.testing.assert_array_equal(RectangleGate(["X"], [5], [None]).evaluate(acc), [False, False, True, True])
    np.testing.assert_array_equal(RectangleGate(["X"], [None], [None]).evaluate(acc), [True] * 4)


def test_rectangle_2d():
    acc = FakeAccess({"X": [1, 5, 5, 10, 5], "Y": [1, 5, 20, 5, 10]})
    g = RectangleGate(["X", "Y"], [2, 2], [10, 10])
    np.testing.assert_array_equal(g.evaluate(acc), [False, True, False, False, False])
    g2 = RectangleGate(["X", "Y"], [2, None], [None, 10])
    np.testing.assert_array_equal(g2.evaluate(acc), [False, True, False, True, False])
    assert not g.is_range


def test_rectangle_missing_param_raises_keyerror():
    with pytest.raises(KeyError):
        RectangleGate(["Z"], [0], [1]).evaluate(FakeAccess({"X": [1]}))


# ---------------------------------------------------------------------- polygon
def test_polygon_identity_scale():
    acc = FakeAccess({"X": [0.5, 0.9, 0.2], "Y": [0.5, 0.9, 0.1]})
    g = PolygonGate(["X", "Y"], [(0.1, 0.1), (0.8, 0.1), (0.8, 0.8), (0.1, 0.8)])
    np.testing.assert_array_equal(g.evaluate(acc), [True, False, True])


def test_polygon_evaluated_in_display_space():
    # Under a 4-decade log scale (1..1e4) the triangle (10,10),(1000,10),(10,1000)
    # becomes (0.25,0.25),(0.75,0.25),(0.25,0.75) on screen.
    scales = {"X": LogScale(1, 1e4), "Y": LogScale(1, 1e4)}
    # (50, 50) -> (0.425, 0.425) and (20, 300) -> (0.325, 0.619) are inside
    acc = FakeAccess({"X": [400, 50, 20, 5000], "Y": [400, 50, 300, 20]}, scales)
    g = PolygonGate(["X", "Y"], [(10, 10), (1000, 10), (10, 1000)])
    disp = g.display_vertices(acc)
    np.testing.assert_allclose(disp, [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75)])
    # (400, 400) would be inside the linear triangle (400 + 400 < 1010) but
    # is at (0.65, 0.65) on screen, which is outside.
    np.testing.assert_array_equal(g.evaluate(acc), [False, True, True, False])


# ---------------------------------------------------------------------- ellipse
def test_ellipse_axis_aligned():
    acc = FakeAccess({"X": [0, 1.9, 0, 1.5, 2.1], "Y": [0, 0, 1.1, 0.6, 0]})
    g = EllipseGate(["X", "Y"], (0, 0), (2, 0), (0, 1))
    np.testing.assert_array_equal(g.evaluate(acc), [True, True, False, True, False])


def test_ellipse_rotated():
    acc = FakeAccess({"X": [0.9, 0.6, 0.3, -0.9], "Y": [0.9, -0.6, -0.3, -0.9]})
    g = EllipseGate(["X", "Y"], (0, 0), (1, 1), (-0.5, 0.5))
    np.testing.assert_array_equal(g.evaluate(acc), [True, False, True, True])


def test_ellipse_degenerate():
    acc = FakeAccess({"X": [0, 1], "Y": [0, 1]})
    g = EllipseGate(["X", "Y"], (0, 0), (1, 1), (2, 2))
    assert not g.evaluate(acc).any()


def test_ellipse_points_inside_static():
    c, a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0]), np.array([0.0, 1.0])
    out = EllipseGate.points_inside(np.array([0.5, 1.5]), np.array([0.5, 0.0]), c, a, b)
    np.testing.assert_array_equal(out, [True, False])


# --------------------------------------------------------------------- negation
@pytest.mark.parametrize(
    "gate",
    [
        RectangleGate(["X", "Y"], [0.2, 0.2], [0.6, 0.6]),
        PolygonGate(["X", "Y"], [(0.2, 0.2), (0.6, 0.2), (0.6, 0.6), (0.2, 0.6)]),
        EllipseGate(["X", "Y"], (0.4, 0.4), (0.6, 0.4), (0.4, 0.6)),
    ],
)
def test_negated_membership(gate):
    acc = FakeAccess({"X": [0.4, 0.9, 0.05], "Y": [0.4, 0.9, 0.5]})
    inside = gate.evaluate(acc)
    np.testing.assert_array_equal(inside, [True, False, False])
    np.testing.assert_array_equal(gate.membership(acc), inside)
    gate.negated = True
    np.testing.assert_array_equal(gate.membership(acc), [False, True, True])
    # evaluate() ignores the flag
    np.testing.assert_array_equal(gate.evaluate(acc), inside)


# ---------------------------------------------------------------------- boolean
def test_boolean_gates():
    masks = {"/A": [True, True, False, False], "/B": [True, False, True, False]}
    acc = FakeAccess({"X": [0, 0, 0, 0]}, masks=masks)
    np.testing.assert_array_equal(BooleanGate("and", ["/A", "/B"]).evaluate(acc), [True, False, False, False])
    np.testing.assert_array_equal(BooleanGate("OR", ["/A", "/B"]).evaluate(acc), [True, True, True, False])
    np.testing.assert_array_equal(BooleanGate("not", ["/A"]).evaluate(acc), [False, False, True, True])
    # unresolved references count as empty
    np.testing.assert_array_equal(BooleanGate("or", ["/A", "/missing"]).evaluate(acc), masks["/A"])
    np.testing.assert_array_equal(BooleanGate("and", ["/A", "/missing"]).evaluate(acc), [False] * 4)
    assert not BooleanGate("and", []).evaluate(acc).any()
    # combining does not mutate the referenced masks
    g = BooleanGate("or", ["/A", "/B"])
    a_before = acc.resolve("/A").copy()
    g.evaluate(acc)
    np.testing.assert_array_equal(acc.resolve("/A"), a_before)


def test_boolean_describe():
    assert BooleanGate("and", ["/Lymph/CD3", "/Lymph/CD4"]).describe() == "CD3 AND CD4"
    assert BooleanGate("or", ["/A", "/B", "/C"]).describe() == "A OR B OR C"
    assert BooleanGate("not", ["/X/Y"]).describe() == "NOT Y"
    assert BooleanGate("not", []).describe() == "NOT"
    assert BooleanGate("AND", ["/A"]).op == "and"


# ---------------------------------------------------------------------- quadrants
def test_quadrant_names_and_geometry():
    quads = make_quadrant_gates("FITC-A", "PE-A", 1000, 2000, xlabel="CD3", ylabel="CD4")
    names = [n for n, _ in quads]
    assert names == ["Q1: CD3- , CD4+", "Q2: CD3+ , CD4+", "Q3: CD3+ , CD4-", "Q4: CD3- , CD4-"]
    groups = {g.quadrant[0] for _, g in quads}
    assert len(groups) == 1
    assert [g.quadrant[1] for _, g in quads] == [1, 2, 3, 4]
    acc = FakeAccess({"FITC-A": [0, 5000, 5000, 0], "PE-A": [5000, 5000, 0, 0]})
    for k, (_, g) in enumerate(quads):
        expected = np.zeros(4, dtype=bool)
        expected[k] = True
        np.testing.assert_array_equal(g.evaluate(acc), expected)
    assert make_quadrant_gates("X", "Y", 0, 0)[0][0] == "Q1: X- , Y+"


def test_quadrants_partition_the_plane():
    rng = np.random.default_rng(3)
    x = np.concatenate([rng.normal(0, 3000, 5000), [1000, 1000, -1e9, 1e9]])
    y = np.concatenate([rng.normal(0, 3000, 5000), [2000, -5, 2000, 1e9]])
    acc = FakeAccess({"X": x, "Y": y})
    masks = np.array([g.evaluate(acc) for _, g in make_quadrant_gates("X", "Y", 1000, 2000)])
    np.testing.assert_array_equal(masks.sum(axis=0), 1)
    # points exactly on the split go to the upper/right side ([lo, hi) convention)
    assert masks[1, -4]  # (1000, 2000) -> Q2


# -------------------------------------------------------------------- serialization
def _all_gates():
    r = RectangleGate(["A", "B"], [None, 1.5], [10.0, None])
    r.color = "#ff0000"
    r.line_width = 2.5
    r.negated = True
    r.quadrant = ("grp", 3)
    p = PolygonGate(["A", "B"], [(0, 0), (1, 0), (1, 1)])
    e = EllipseGate(["A", "B"], (1, 2), (3, 2), (1, 5))
    b = BooleanGate("or", ["/x", "/y"])
    return [r, p, e, b, RectangleGate(["A"], [0], [1])]


@pytest.mark.parametrize("gate", _all_gates(), ids=lambda g: g.kind)
def test_gate_dict_round_trip(gate):
    d = gate.to_dict()
    again = gate_from_dict(d)
    assert type(again) is type(gate)
    assert again.to_dict() == d
    assert again.id == gate.id
    clone = gate.clone()
    assert clone.id != gate.id
    cd = clone.to_dict()
    cd.pop("id")
    d2 = dict(d)
    d2.pop("id")
    assert cd == d2


def test_gate_dict_details():
    r = _all_gates()[0]
    d = r.to_dict()
    assert d["kind"] == "rectangle"
    assert d["lo"] == [None, 1.5] and d["hi"] == [10.0, None]
    assert d["negated"] is True and d["color"] == "#ff0000" and d["line_width"] == 2.5
    assert d["quadrant"] == ["grp", 3]
    again = gate_from_dict(d)
    assert again.quadrant == ("grp", 3) and again.negated
    plain = RectangleGate(["A"], [0], [1]).to_dict()
    assert "negated" not in plain and "color" not in plain and "line_width" not in plain
    with pytest.raises(ValueError):
        gate_from_dict({"kind": "hexagon"})


def test_gate_param_helpers():
    g = PolygonGate(["FITC-A", "PE-A"], [(0, 0), (1, 0), (1, 1)])
    assert g.uses_param("PE-A") and not g.uses_param("APC-A")
    g.rename_param("PE-A", "Comp-PE-A")
    assert g.dims == ["FITC-A", "Comp-PE-A"]
    assert not g.is_1d
