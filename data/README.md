# Data directory

The project downloads data from [StatsBomb Open Data](https://github.com/hudl/open-data).
Raw and processed datasets are deliberately ignored by Git.

Expected local layout:

```text
data/
├── raw/statsbomb/
│   ├── matches/43/106.json
│   └── events/<match_id>.json
└── processed/
    └── shots_world_cup_2022.csv
```

When publishing analysis made from these data, credit StatsBomb/Hudl and follow
the attribution terms supplied in the upstream repository.

