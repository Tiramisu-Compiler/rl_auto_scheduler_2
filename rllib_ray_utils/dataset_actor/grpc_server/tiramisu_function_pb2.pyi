from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TiramisuFunctionName(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class TiramisuFuction(_message.Message):
    __slots__ = ["name", "content", "cpp"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    CONTENT_FIELD_NUMBER: _ClassVar[int]
    CPP_FIELD_NUMBER: _ClassVar[int]
    name: str
    content: str
    cpp: str
    def __init__(self, name: _Optional[str] = ..., content: _Optional[str] = ..., cpp: _Optional[str] = ...) -> None: ...
