"""
Run once to populate the section column in studio_manual.
python tag_sections.py
"""
from utils import get_conn

SECTION_RULES = [
    ("cq_%", "FAQ"),
    ("us_%", "User Stories"),
    ("arch/%", "Architecture"),
    ("pitch/%", "Overview"),
]

def tag():
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            for pattern, section in SECTION_RULES:
                cur.execute(
                    "UPDATE studio_manual SET section = %s WHERE id LIKE %s",
                    (section, pattern)
                )
                print(f"  {section}: {cur.rowcount} rows")
            # Everything else
            cur.execute(
                "UPDATE studio_manual SET section = 'General' WHERE section = '' OR section IS NULL"
            )
            print(f"  General: {cur.rowcount} rows")
        conn.commit()
        print("Done.")
    finally:
        conn.close()

if __name__ == "__main__":
    tag()
