from cardassist.db import connect
from data.generate_data import build


def test_row_counts(db):
    with connect(db) as c:
        assert c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 200
        assert c.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 10


def test_generation_is_deterministic(tmp_path):
    a, b = build(tmp_path / "a.db"), build(tmp_path / "b.db")
    q = "SELECT * FROM transactions ORDER BY transaction_id"
    with connect(a) as ca, connect(b) as cb:
        assert [tuple(r) for r in ca.execute(q)] == [tuple(r) for r in cb.execute(q)]
