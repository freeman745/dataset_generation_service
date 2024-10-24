from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class MethodLog:
    cls_name: str
    method_name: str
    start_time: datetime
    end_time: datetime
    duration: float

    @property
    def name(self):
        return f"{self.cls_name}.{self.method_name}"

    def __repr__(self) -> str:
        desc = f"{self.name} ({int(self.duration * 1000)} ms)"
        return desc


@dataclass(frozen=True, eq=True)
class ActionLog:
    name: str
    action_id: str
    res: dict = field(compare=False)
    start_time: datetime
    end_time: datetime
    duration: float

    def __repr__(self) -> str:
        desc = (
            f"{self.__class__.__name__}"
            f"({self.full_name}, {int(self.duration * 1000)} ms"
        )
        desc += ", Success" if self.is_successful else ", Fail"
        desc += ")"
        return desc

    @property
    def step(self) -> Optional[str]:
        try:
            return self.action_id.split("_")[2]
        except:
            return None

    @property
    def full_name(self):
        return f"{self.step}.{self.name}"

    @property
    def is_successful(self) -> bool:
        return (
            self.res["error_id"] >= 0
            or self.res["error_id"] == -5  # FinishedNoMoreDest
        )


@dataclass(frozen=True, eq=True)
class StepLog:
    name: str
    start_time: datetime
    end_time: datetime
    duration: float
    status: str

    def __repr__(self) -> str:
        desc = (
            f"{self.__class__.__name__}({self.name}, {int(self.duration * 1000)} ms, "
        )
        desc += "Success" if self.is_successful else "Fail"
        desc += ")"
        return desc

    @property
    def is_successful(self) -> bool:
        return self.status == "ORDER_STEP_OK"
