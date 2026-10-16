"""Narrow CustomAlbums profile, verified against pinned template and source.

This is format evidence, not a successful game import certificate.
"""
from dataclasses import dataclass
from mdac2.domain.chart import ChartEvent


@dataclass(frozen=True)
class BasicProfile:
    profile_id: str = "customalbums-bdfc7e6-basic-v1"
    loader_commit: str = "bdfc7e6c2f4be0c8b5c5b405ba26b287397f956f"
    template_commit: str = "ea3585b4f95b824a911450546aa08e4d44814409"
    game_verified: bool = False
    max_cells: int = 16384
    max_measure: int = 999

    def objects(self, event: ChartEvent):
        if event.accent:
            raise ValueError("Accent mapping is not verified by this profile")
        if not event.lane_mask or len(set(event.lane_mask)) != len(event.lane_mask) or not set(event.lane_mask) <= {"ground", "air"}:
            raise ValueError("Invalid lanes")
        if event.semantic == "gemini":
            if set(event.lane_mask) != {"ground", "air"} or event.duration_q:
                raise ValueError("Gemini requires two lanes and zero duration")
            return (("13", "0E"), ("14", "0E"))
        if len(event.lane_mask) != 1:
            raise ValueError("Non-Gemini requires one physical lane in this profile")
        lane = event.lane_mask[0]
        if event.semantic == "hold" and event.duration_q > 0:
            return (("53" if lane == "air" else "54", "0F"),)
        if event.semantic == "tap" and event.duration_q == 0:
            return (("13" if lane == "air" else "14", "01"),)
        raise ValueError("Unsupported event semantics/duration")


BASIC = BasicProfile()
