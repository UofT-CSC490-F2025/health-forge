import psycopg2
from psycopg2.extras import RealDictCursor

HOST = "terraform-20251109014506835900000003.cjac4g4syo66.us-east-2.rds.amazonaws.com"
USER = "healthforgedb"
PASSWORD = "mimicpassword"

TARGET_DB = "raw_training_data"  # the actual dataset DB


def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80 + "\n")


# -------------------------------
# Step 1: Show all databases
# -------------------------------
def list_databases():
    conn = psycopg2.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        dbname="postgres",
        sslmode="require"
    )
    cur = conn.cursor()
    cur.execute("SELECT datname FROM pg_database;")
    dbs = cur.fetchall()
    print_section("Databases on this RDS instance:")
    for db in dbs:
        print(" -", db[0])
    cur.close()
    conn.close()


# -------------------------------
# Step 2: Inspect the target DB
# -------------------------------
def inspect_database():
    conn = psycopg2.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        dbname=TARGET_DB,
        sslmode="require"
    )
    cur = conn.cursor()

    # ---- schemas ----
    print_section("Schemas in database:")
    cur.execute("""
        SELECT schema_name 
        FROM information_schema.schemata
        WHERE schema_name NOT IN ('pg_catalog', 'information_schema');
    """)
    schemas = cur.fetchall()
    for schema in schemas:
        print(" -", schema[0])

    # ---- tables ----
    print_section("Tables in database:")
    cur.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name;
    """)
    tables = cur.fetchall()
    if not tables:
        print("No tables found.")
        return

    for schema, table in tables:
        print(f"{schema}.{table}")

    # ---- row counts ----
    print_section("Row counts per table:")
    for schema, table in tables:
        cur.execute(f"SELECT COUNT(*) FROM {schema}.{table};")
        count = cur.fetchone()[0]
        print(f"{schema}.{table}: {count} rows")

    # ---- columns ----
    print_section("Columns for each table:")
    for schema, table in tables:
        print(f"\nTable: {schema}.{table}")
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema=%s AND table_name=%s;
        """, (schema, table))
        cols = cur.fetchall()
        for col in cols:
            print(f" - {col[0]} ({col[1]})")

    # ---- sample rows ----
    print_section("Sample rows (up to 5 per table):")
    detailed_cur = conn.cursor(cursor_factory=RealDictCursor)

    for schema, table in tables:
        print(f"\n---- {schema}.{table} ----")
        detailed_cur.execute(f"SELECT * FROM {schema}.{table} LIMIT 5;")
        rows = detailed_cur.fetchall()
        if not rows:
            print("No rows.")
        else:
            for r in rows:
                print(r)

    cur.close()
    conn.close()
    print("\n\nFinished database inspection.")


# -------------------------------
# Run everything
# -------------------------------
if __name__ == "__main__":
    list_databases()
    inspect_database()
