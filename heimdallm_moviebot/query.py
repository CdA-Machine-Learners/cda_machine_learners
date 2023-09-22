import time
from datetime import date
from typing import Callable

from heimdallm.bifrosts.sql.postgres.select.bifrost import Bifrost


class Result:
    """Represents the results and query metadata"""

    def __init__(self, *, query, columns, rows):
        self.query = query
        self.columns = columns
        self.rows = rows


def build_query_executer(make_conn, bifrost: Bifrost) -> Callable[[str], Result]:
    def query(untrusted_input: str) -> Result:
        """run a natural language query against the db"""

        trusted_sql_query = bifrost.traverse(untrusted_input)
        conn = make_conn()
        try:
            res = _exec_query(conn, trusted_sql_query)
        finally:
            conn.close()
        return res

    return query


def _exec_query(conn, trusted_sql_query: str, **params) -> Result:
    now = time.mktime(date.today().timetuple())
    all_params = {"timestamp": now}
    all_params.update(params)

    cur = conn.cursor()
    cur.execute(trusted_sql_query, all_params)
    columns = [desc[0] for desc in cur.description]

    rows = cur.fetchall()
    res = Result(
        query=trusted_sql_query,
        columns=columns,
        rows=rows,
    )
    return res
