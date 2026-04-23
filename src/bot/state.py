from dataclasses import dataclass, field

from .config import MAX_HISTORY


@dataclass
class BotState:
    sessions: dict[int, list[dict[str, str]]] = field(default_factory=dict)
    active_models: dict[int, str] = field(default_factory=dict)
    user_modes: dict[int, str] = field(default_factory=dict)
    user_models: dict[int, str] = field(default_factory=dict)

    def update_history(self, chat_id: int, role: str, text: str) -> None:
        if chat_id not in self.sessions:
            self.sessions[chat_id] = []
        self.sessions[chat_id].append({"role": role, "text": text})
        if len(self.sessions[chat_id]) > MAX_HISTORY:
            self.sessions[chat_id] = self.sessions[chat_id][-MAX_HISTORY:]

    def clear_history(self, chat_id: int) -> None:
        self.sessions.pop(chat_id, None)
        self.active_models.pop(chat_id, None)


state = BotState()
