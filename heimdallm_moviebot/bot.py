import os
from typing import cast

import discord, settings
from discord import app_commands
from dotenv import load_dotenv
from heimdallm.bifrosts.sql import exc
from heimdallm.llm_providers import openai

import movies
import query
import settings

load_dotenv()

CDA_GUILD = discord.Object(id=settings.SERVER_ID)


class DiscordClient(discord.Client):
    def __init__(self):
        intents = discord.Intents.default()
        super().__init__(intents=intents)
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self):
        # manually sync over the guild commands so they appear immediately
        self.tree.copy_global_to(guild=CDA_GUILD)
        await self.tree.sync(guild=CDA_GUILD)


movie_client = DiscordClient()


@movie_client.event
async def on_ready():
    print(f"Logged in as {movie_client.user} (ID: {movie_client.user.id})")
    print("------")


# openai will compose the query
llm = openai.Client(api_key=settings.OPENAI_API_KEY, model=settings.MODEL)

# the bifrost is a callable that can produce sql queries from natural language
bifrost = movies.build_bifrost(llm)

# the query executer completes the full pipeline from natural language to db results
query_executer = query.build_query_executer(movies.make_conn, bifrost)


@movie_client.tree.command(name="movie-query")
@app_commands.rename(nl_query="query")
@app_commands.describe(nl_query="The natural language movie query")
async def movie_query(interaction: discord.Interaction, nl_query: str):
    """Queries the movie database with a natural language query."""
    await interaction.response.defer()

    try:
        res: query.Result = query_executer(nl_query)
    except exc.BaseException as e:
        metadata_embed = discord.Embed(
            title="🚫 Validation Error",
            color=discord.Colour.red(),
        )
        metadata_embed.add_field(
            name="✍️ Natural language query",
            value=f'"{nl_query}"',
            inline=False,
        )
        metadata_embed.add_field(
            name="🔍 Exception",
            value=f"```{e.__class__.__name__}: {e}```",
            inline=False,
        )
        metadata_embed.add_field(
            name="🤨 Untrusted LLM sql query",
            value=f"```sql\n{e.ctx.untrusted_llm_output}```",
            inline=False,
        )

        await interaction.followup.send(
            "Your query could not be validated 😭",
            embed=metadata_embed,
        )
        return
    except Exception as e:
        metadata_embed = discord.Embed(
            title="🚫 Query execution error",
            color=discord.Colour.red(),
        )
        metadata_embed.add_field(
            name="✍️ Natural language query",
            value=f'"{nl_query}"',
            inline=False,
        )
        metadata_embed.add_field(
            name="🔍 Exception",
            value=f"```{e.__class__.__name__}: {e}```",
            inline=False,
        )

        await interaction.followup.send(
            "Your query could not be executed 😱",
            embed=metadata_embed,
        )
        return

    metadata_embed = discord.Embed(
        title="ℹ️ Query Information",
        color=discord.Colour.blue(),
    )
    metadata_embed.add_field(
        name="✍️ Natural language query",
        value=f'"{nl_query}"',
        inline=False,
    )
    metadata_embed.add_field(
        name="✅ Validated LLM sql query",
        value=f"```sql\n{res.query}```",
        inline=False,
    )

    res_embed = discord.Embed(
        title="🎉 Query Results",
        color=discord.Colour.green(),
    )
    if res.rows:
        for i, row in enumerate(res.rows):
            col_row = zip(res.columns, row)
            fields = []
            for col_name, col_val in col_row:
                if col_name.endswith("imdb") and col_val:
                    col_name = "imdb"
                    col_val = f"[link](https://www.imdb.com/title/{col_val}/)"
                else:
                    col_val = f"`{col_val}`"
                fields.append(f"{col_name}: {col_val}")
            res_embed.add_field(
                name=f"Row {i+1}",
                value="\n".join(fields),
                inline=False,
            )
    else:
        res_embed.add_field(
            name="No results",
            value="Your query returned no results 😭",
            inline=False,
        )

    await interaction.followup.send(
        embeds=[metadata_embed, res_embed],
    )


movie_client.run(cast(str, settings.DISCORD_TOKEN))
