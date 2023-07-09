import os
import subprocess

import click
from tqdm.auto import tqdm


@click.command()
@click.option("--games", default=8192, help="Number of games to play and track.")
@click.option("--folder", default="./games/", help="Output filename")
@click.option("--batch-size", default=4096, help="Batch size for generating games.")
@click.option("--prefix", default="", help="Prefix to append before the filename of the generated games.")
def generate_games_batched(games: int, folder: str, batch_size: int, prefix: str) -> None:
    if os.path.exists(folder):
        response = input(f"Folder {folder} already exists, do you want to continue? y, n\n")
        if response.lower() != "y":
            print("Quitting")
            return
    else:
        os.mkdir(folder)

    print(f"Saving games to separate .parquet files in {folder}")

    prefix = f"{prefix}_" if prefix != "" else ""
    games_generated = 0
    n_batches = games // batch_size + int(games % batch_size > 0)
    for batch_index in tqdm(range(n_batches), f"Generating {games} games in batches of {batch_size} games"):
        # Generate games in batches
        remaining_games = games - games_generated
        games_to_generate = min(batch_size, remaining_games)

        fname = os.path.join(folder, f"{prefix}{batch_index}.parquet")
        process = subprocess.Popen(f"python generate_games.py --games={games_to_generate} --fname={fname}",
                                   shell=True,
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        process.wait()

        games_generated += games_to_generate

    print(f"Finished generating {games_generated} games. Saved games to separate .parquet files in {folder}")


if __name__ == "__main__":
    generate_games_batched()
