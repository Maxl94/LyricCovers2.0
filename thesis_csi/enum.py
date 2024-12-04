from enum import Enum


class Features(Enum):
    SONG_TEXT = "song_text"
    LYRICS = "lyrics"
    TRANSCRIPTION = "transcription"
    VOCALS = "vocals"

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        member.column = value
        return member


class Labels(Enum):
    ORIGINAL_SONG_ID = "original_id"
    LABEL = "label"

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        member.column = value
        return member


class Metadata(Enum):
    SONG_ID = "id"
    SONG_TEXT_TYPE = "song_text_type"
    IS_COVER = "is_cover"
    TRANSCRIPTION_STATUS = "transcription_status"
    SOURCE_SEPARATION_STATUS = "source_separation_status"

    def __new__(cls, value):
        member = object.__new__(cls)
        member._value_ = value
        member.column = value
        return member
