# traj_lib/utils/register.py

from typing import Any, Callable, Dict, Iterator, List
from collections.abc import MutableMapping
from collections import defaultdict
import re
from difflib import get_close_matches


def _normalize(s: str) -> str:
    return re.sub(r'[^a-z0-9]', '', str(s).casefold())


class CategoryRegistry(MutableMapping[str, Any]):

    def __init__(self, category: str) -> None:
        self.category = category
        self._data: Dict[str, Any] = {}

    def __getitem__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError:                    
            suggestions = self.suggest(key, n=3, cutoff=0.6)
            if suggestions:
                msg = (
                    f"{key!r} not registered in {self.category!r}. "
                    f"Did you mean: {', '.join(suggestions)} ?"
                )
            else:
                msg = (
                    f"{key!r} not registered in {self.category!r}. "
                    f"Available: {', '.join(sorted(self._data.keys()))}"
                )
            raise KeyError(msg) from None

    def __setitem__(self, key: str, value: Any) -> None:
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"<CategoryRegistry {self.category!r} keys={list(self._data.keys())!r}>"

    # --- 额外能力 ---
    def register(self, name: str) -> Callable[[Any], Any]:
        def decorator(obj: Any) -> Any:
            if name in self._data:
                return self._data[name]
            self._data[name] = obj
            return obj
        return decorator

    def suggest(self, target: str, n: int = 3, cutoff: float = 0.6) -> List[str]:
        norm_to_raw: Dict[str, List[str]] = {}
        for k in self._data.keys():
            nk = _normalize(k)
            norm_to_raw.setdefault(nk, []).append(k)

        target_norm = _normalize(target)
        match_norms = get_close_matches(target_norm, list(norm_to_raw.keys()), n=n, cutoff=cutoff)

        seen, out = set(), []
        for nk in match_norms:
            for raw in norm_to_raw[nk]:
                if raw not in seen:
                    seen.add(raw)
                    out.append(raw)
                    if len(out) >= n:
                        break
            if len(out) >= n:
                break
        return out


class RegistryHub:

    def __init__(self) -> None:
        self._cats: Dict[str, CategoryRegistry] = {}

    def category(self, name: str) -> CategoryRegistry:
        if name not in self._cats:
            self._cats[name] = CategoryRegistry(name)
        return self._cats[name]

    def register(self, category: str, name: str) -> Callable[[Any], Any]:
        return self.category(category).register(name)

    def __getitem__(self, category: str) -> CategoryRegistry:
        return self.category(category)

_HUB = RegistryHub()

DATALOADER_REGISTRY: CategoryRegistry = _HUB.category("dataloader")
EVAL_REGISTRY: CategoryRegistry       = _HUB.category("eval")
VIEW_REGISTRY: CategoryRegistry       = _HUB.category("view")

def register_dataloader(name: str):
    return DATALOADER_REGISTRY.register(name)

def register_eval(name: str):
    return EVAL_REGISTRY.register(name)

def register_view(name: str):
    return VIEW_REGISTRY.register(name)

def _register(category: str, name: str) -> Callable[[Any], Any]:
    return _HUB.register(category, name)

_REGISTRY: Dict[str, Dict[str, Any]] = defaultdict(dict)
_REGISTRY["dataloader"] = DATALOADER_REGISTRY._data
_REGISTRY["eval"]       = EVAL_REGISTRY._data
_REGISTRY["view"]       = VIEW_REGISTRY._data
