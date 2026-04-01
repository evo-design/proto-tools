# Beta Tester Instructions

[![Discord](https://img.shields.io/badge/Discord-Join-5865F2?logo=discord&logoColor=white)](https://discord.gg/evs3Unkegv)

Welcome! Thank you for taking an early look at our codebase. We are really excited to release this project and get it in the hands of researchers.

Please explore as much or as little as you want. We would love to hear any feedback or suggestions you have. Feel free to open issues and/or pull requests.

## Getting Started

1. Follow the installation steps in the [README](../README.md). Make sure you have the `PROTO_HOME` and `PROTO_MODEL_CACHE` environment variables configured in your `~/.bashrc` where you would like them to be set to avoid any issues with storage limitations on your machine! (see [Model Weights](model-weights.md) for more details).

2. Try running some tools and let us know your thoughts!


3. **(Optional)** One thing that would really help us is understanding how well our tool dependency isolation system generalizes to other clusters. If you are willing, please run `pytest --env-report`. This recreates all tool environments, runs tests, and generates a compatibility report for your platform. Note that this takes around 3 hours to complete.

## Permissions

As a beta tester, you have **Triage** permissions on this repo. This means you can:

- View and clone the repository
- Open and manage issues
- Comment on pull requests
- Create private forks

If you'd like to contribute code, you can create a **private fork** of the repo and submit a pull request from there.

## Feedback

We would love to hear from you! If you run into any issues or have suggestions:

- Send us a message in Discord
- Open an issue on this repo with details on what you were trying to do, what happened, and any error messages you encountered

## Reminders

- This is pre-release software; expect rough edges!
- Please do not share the source code externally


# Thank you!
Team Proto
