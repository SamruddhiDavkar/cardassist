"""Generate fake customers and ~200 transactions into SQLite.

Run:  python -m data.generate_data            (writes data/cardassist.db)

Customer C001 gets a set of planted transactions that the evaluation set relies on
(boundary cases for the 60-day window and the $500 review threshold, a duplicate charge,
a pending charge, an already-disputed charge, and so on). Everything else is random but
seeded, so the database is identical on every run.
"""
import random
import sys
from datetime import timedelta
from pathlib import Path

from faker import Faker

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cardassist import config  # noqa: E402
from cardassist.db import connect, init_schema  # noqa: E402

SEED = 42
N_CUSTOMERS = 10
N_TRANSACTIONS = 200

MERCHANTS = [
    ("Green Basket Grocery", "groceries", (12, 160)),
    ("FuelUp Station", "gas", (25, 75)),
    ("Corner Bistro", "dining", (15, 90)),
    ("PageTurner Books", "shopping", (8, 60)),
    ("CityTransit", "transport", (2, 40)),
    ("CloudBox Storage", "subscription", (2.99, 9.99)),
    ("Pharma Plus", "health", (5, 80)),
    ("Urban Threads", "shopping", (20, 220)),
    ("SoundWave Music", "subscription", (10.99, 10.99)),
    ("Metro Cinema", "entertainment", (12, 45)),
]

D = lambda n: (config.TODAY - timedelta(days=n)).isoformat()  # noqa: E731

# (transaction_id, days_ago, merchant, category, amount, status)
PLANTED_C001 = [
    ("T9001", 8,  "StreamFlix", "subscription", 49.99, "posted"),
    ("T9002", 5,  "BestTech Store", "electronics", 849.00, "posted"),
    ("T9003", 75, "Coastal Airlines", "travel", 312.40, "posted"),
    ("T9004", 3,  "Daily Grind Cafe", "dining", 6.75, "posted"),
    ("T9005", 3,  "Daily Grind Cafe", "dining", 6.75, "posted"),   # duplicate
    ("T9006", 20, "QuickMart", "groceries", 23.10, "disputed"),
    ("T9007", 0,  "RideNow", "transport", 18.40, "pending"),
    ("T9008", 2,  "Lux Watches Online", "shopping", 299.00, "posted"),
    ("T9009", 10, "HomeGoods Depot", "home", 500.00, "posted"),     # exactly $500
    ("T9010", 60, "BookNook", "shopping", 27.50, "posted"),          # exactly 60 days
    ("T9011", 61, "PetPals", "pets", 64.20, "posted"),               # 61 days
    ("T9012", 20, "GymPro", "fitness", 39.99, "posted"),             # Oct 1
    ("T9013", 50, "GymPro", "fitness", 39.99, "posted"),             # Sep 1
    ("T9014", 81, "GymPro", "fitness", 39.99, "posted"),             # Aug 1
]


def build(db_path: Path | None = None) -> Path:
    db_path = Path(db_path or config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    fake = Faker()
    Faker.seed(SEED)
    rng = random.Random(SEED)

    with connect(db_path) as conn:
        init_schema(conn)
        customers = []
        for i in range(1, N_CUSTOMERS + 1):
            cid = f"C{i:03d}"
            customers.append(cid)
            conn.execute(
                "INSERT INTO customers VALUES (?,?,?,?)",
                (cid, fake.name(), fake.email(), fake.credit_card_number(card_type="visa16")),
            )

        for tid, days, merchant, cat, amount, status in PLANTED_C001:
            conn.execute(
                "INSERT INTO transactions VALUES (?,?,?,?,?,?,?)",
                (tid, "C001", D(days), merchant, cat, amount, status),
            )

        for n in range(1, N_TRANSACTIONS - len(PLANTED_C001) + 1):
            merchant, cat, (lo, hi) = rng.choice(MERCHANTS)
            days = rng.randint(1, 120)
            status = "pending" if days <= 1 else "posted"
            conn.execute(
                "INSERT INTO transactions VALUES (?,?,?,?,?,?,?)",
                (f"T{n:04d}", rng.choice(customers), D(days), merchant, cat,
                 round(rng.uniform(lo, hi), 2), status),
            )
    return db_path


if __name__ == "__main__":
    path = build()
    with connect(path) as c:
        n = c.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    print(f"Wrote {n} transactions for {N_CUSTOMERS} customers to {path}")
