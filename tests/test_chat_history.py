from src.llm_service import format_chat_history


def test_format_chat_history_includes_recent_messages() -> None:
    formatted = format_chat_history(
        [
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one"},
            {"role": "user", "content": "Question two"},
        ]
    )

    assert "User: Question one" in formatted
    assert "Assistant: Answer one" in formatted
    assert "User: Question two" in formatted
