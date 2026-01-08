from typing import TYPE_CHECKING, Any, TypeAlias

if TYPE_CHECKING:
    import pandas as pd
    import polars as pl
    import torch
    import torch_geometric.data.data as pyg_Data

    DataFrame: TypeAlias = pl.DataFrame
    Tensor: TypeAlias = torch.Tensor
    PyGData: TypeAlias = pyg_Data.Data

else:
    DataFrame = Any
    Tensor = Any
    PyGData = Any

__all__ = ["DataFrame", "Tensor", "PyGData"]
