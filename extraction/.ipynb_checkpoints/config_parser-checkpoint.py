import argparse
import json
import re


def get_clean_instructions(config_data):
    """Extract and clean the game description."""
    curriculum = config_data.get("curriculum", [])

    for phase in curriculum:
        if phase.get("type") == "message" and "text" in phase:
            text_data = phase["text"]

            if isinstance(text_data, list):
                cleaned_lines = [
                    line
                    for line in text_data
                    if not ("(press" in line.lower() and "start" in line.lower())
                ]
                full_text = "\n".join(cleaned_lines).strip()
            else:
                full_text = str(text_data).strip()

            full_text = re.split(
                r"\n?Controls:",
                full_text,
                flags=re.IGNORECASE,
            )[0].strip()

            return full_text

    return "No message/instruction phase found."


def get_available_actions(config_data):
    """Extract available actions from the game phase."""
    curriculum = config_data.get("curriculum", [])

    for phase in curriculum:
        if phase.get("type") == "game" and "keys" in phase:
            return ", ".join(phase["keys"].keys())

    return "No actions found."


def main():
    parser = argparse.ArgumentParser(
        description="Extract game instructions and available actions from a config."
    )
    parser.add_argument(
        "config",
        help="Path to the game config JSON file.",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_data = json.load(f)

    print("Instructions:")
    print(get_clean_instructions(config_data))

    print("\nAvailable actions:")
    print(get_available_actions(config_data))


if __name__ == "__main__":
    main()