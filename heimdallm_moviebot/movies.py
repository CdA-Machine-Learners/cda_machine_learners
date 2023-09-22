import os
from typing import Sequence

import jinja2
import psycopg2
from heimdallm.bifrosts.sql.common import (
    ANY_JOIN,
    FqColumn,
    JoinCondition,
    ParameterizedConstraint,
)
from heimdallm.bifrosts.sql.postgres.select.bifrost import Bifrost
from heimdallm.bifrosts.sql.postgres.select.envelope import (
    PromptEnvelope as _PromptEnvelope,
)
from heimdallm.bifrosts.sql.postgres.select.validator import (
    ConstraintValidator as _ConstraintValidator,
)
from heimdallm.llm import LLMIntegration


def make_conn():
    """returns a connection to the movie database"""
    conn = psycopg2.connect(
        host="localhost",
        dbname="movies",
        user="postgres",
        password=os.getenv("POSTGRES_PASSWORD"),
    )
    return conn


class ConstraintValidator(_ConstraintValidator):
    """Basic constraints on the query. Currently only restricts to non-id
    columns and adds a limit."""

    def requester_identities(self) -> Sequence[ParameterizedConstraint]:
        return []

    def parameterized_constraints(self) -> Sequence[ParameterizedConstraint]:
        return []

    def condition_column_allowed(self, fq_column: FqColumn) -> bool:
        return True

    def select_column_allowed(self, column: FqColumn) -> bool:
        is_id = column.name.endswith("_id") or column.column == "id"
        return not is_id

    def allowed_joins(self) -> Sequence[JoinCondition]:
        return [ANY_JOIN]

    def max_limit(self) -> int | None:
        return 5


class PromptEnvelope(_PromptEnvelope):
    def template(self, env: jinja2.Environment) -> jinja2.Template:
        with open("movie-prompt.j2", "r") as h:
            return env.from_string(h.read())


def build_bifrost(llm: LLMIntegration) -> Bifrost:
    """build the thing that will produce sql queries from natural language"""

    # we'll use the movie database schema for context
    db_schema = open("movie-schema.sql", "r").read()

    validator = ConstraintValidator()
    validators = [validator]

    envelope = PromptEnvelope(
        llm=llm,
        db_schema=db_schema,
        validators=validators,
    )

    bifrost = Bifrost(
        prompt_envelope=envelope,
        llm=llm,
        constraint_validators=validators,
    )
    return bifrost
