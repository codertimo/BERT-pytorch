# REPO-README

Template for my python projects

## Template Replace Check-List

- [ ] Make your own package name 👋
- [ ] Replace `package/` to new package name 🎉
- [ ] Replace command in `.github/workflows/main.yml` with new package name 🔨
- [ ] Replace command in `Makefile` with new package name
- [ ] Replace name, description, author etc in `setup.py` with new package setting 🏄‍♂️
- [ ] Replace author, version in `package/__init__.py` to new package name
- [ ] Setting codecov (https://docs.codecov.io/docs/quick-start) to your repo
- [ ] Make REAL runnable code 👨‍💻
- [ ] Make REAL test code 👩🏻‍💻
- [ ] Remove this README and make your own story! 👍

## Run Scripts

All runnable python scripts should be located in `scripts/` folder

And you can run the scripts through under command

```shell
python -m scripts.run_sample
```

## Run Linting

This project use three Linter: `black`, `isort`, `flake8`

```
# use linter to fix code format
make style

# check lint error
make quality
```

## Run Test

All runnable test codes should be located in `tests/` folder

```shell
pytest
```

## Author

by @codertimo
