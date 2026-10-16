# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the CC BY-NC 4.0 license found in the
# LICENSE file in the root directory of this source tree.


import dataclasses
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd


@dataclasses.dataclass
class CSVLogger:
    filename: Union[str, Path]
    fields: Optional[List[str]] = None

    def _ensure_fields(self, keys: Iterable[str]) -> None:
        path = Path(self.filename)
        has_data = path.exists() and path.stat().st_size > 0
        if self.fields is None:
            self.fields = list(pd.read_csv(path, nrows=0).columns) if has_data else sorted(keys)
        added = [key for key in sorted(keys) if key not in self.fields]
        if added:
            self.fields = list(self.fields) + added
            if has_data:
                pd.read_csv(path).reindex(columns=self.fields, fill_value="").to_csv(path, index=False)
                return
        if not has_data:
            pd.DataFrame(columns=self.fields).to_csv(path, index=False)

    def log(self, log_data: Dict[str, Any]) -> None:
        self._ensure_fields(log_data.keys())
        data = {field: log_data.get(field, "") for field in self.fields}  # Ensure all fields are present
        islist = [isinstance(v, Iterable) and not isinstance(v, str) for k, v in data.items()]
        if all(islist):
            df = pd.DataFrame(data)
        elif not any(islist):
            df = pd.DataFrame([data])
        else:
            raise RuntimeError("Fields should all be a numbers, a string or iterable objects. We don't support mixed types.")
        df.to_csv(self.filename, mode="a", header=False, index=False)

    def log_many(self, rows: Iterable[Dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        self._ensure_fields({key for row in rows for key in row})
        data = [{field: row.get(field, "") for field in self.fields} for row in rows]
        pd.DataFrame(data, columns=self.fields).to_csv(self.filename, mode="a", header=False, index=False)
