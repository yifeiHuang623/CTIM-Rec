# CTIM-Rec

A PyTorch-based experimental repository for POI recommendation / next POI prediction. The project currently includes the `CTIM_Rec` model and example datasets for `NYC`, `TKY`, and `CA`.

## Structure

- `main.py`: main entry point
- `model/CTIM_Rec/`: model implementation and configuration
- `data/`: datasets and dataset configs
- `utils/`: utilities for data loading, evaluation, and logging

## Example

```bash
python main.py --model CTIM_Rec --dataset NYC --task NPP
```

Optional arguments:

- `--dataset`: `all` or a specific dataset such as `NYC`
- `--metrics`: `all` or specific evaluation metrics
- `--cfg`: path to an extra YAML config file
- `--task`: task name, default is `NPP`

## Output

Results are saved to `outputs/`, and trained model checkpoints are saved to `saved_models/`.
