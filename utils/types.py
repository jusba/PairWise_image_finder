from typing import TypedDict

class PairRow(TypedDict):
    id_left: str
    date_left: str
    id_right: str
    date_right: str
    left_path: str
    right_path: str