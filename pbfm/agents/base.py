from typing import Any, Literal

import pydantic


class BaseConfig(pydantic.BaseModel):
    """Base class for model configurations."""

    # extra="forbid" -- prevent adding new fields (e.g., accidentally typo-ing argument model-dimm=256 )
    # strict=True -- be strict on types, do not try to cast e.g., "True" --> "1" (or vice versa)
    # use_enum_values=True -- fixes issues with serializing/deserializing enums
    # frozen=True -- make the model immutable
    model_config = pydantic.ConfigDict(extra="forbid", strict=True, use_enum_values=True, frozen=True)

    name: Literal["BaseConfig"] = "BaseConfig"

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # We want to make sure `class_name` is always a valid name.
        # So we will add it to the list of Literals (if user set custom name), or create one.
        # We will default using user's name if one provided, otherwise the class name.
        if not hasattr(cls, "name") or cls.name == "BaseConfig":
            # Add field consistent with `name: Literal["BaseConfig"] = "BaseConfig"`
            cls.name = cls.__name__
            cls.__annotations__["name"] = Literal[cls.__name__]
        else:
            # Append the name to the list of Literals
            cls.__annotations__["name"] = Literal[(cls.__name__,) + cls.__annotations__["name"].__args__]

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key {key} not found in config {self.__class__.__name__}")

    def build(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"The object {self} did not have valid build function. Did you forget to define it?")
