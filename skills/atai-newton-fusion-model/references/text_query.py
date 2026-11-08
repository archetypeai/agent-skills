"""
Text / JSON examples on Model Serving: Newton-C 2.6 (default) or Newton-Fusion 1.0.

Grounded in a realistic domain: a network security analyst reviewing
smart-home WiFi flow logs (the GHOST-IoT activity-detection use case).
Flows are bidirectional conversations between two endpoints with
byte/packet counters, a transport + application protocol, and timestamps.

Demonstrates five patterns:
  1. Plain text Q&A over an inline flow snippet (/inference/query).
  2. Structured JSON output — classify a device (the prompt is the schema).
  3. Reasoning over a longer flow log, inlined in the query. Model Serving
     refuses .txt attachments by every route, so text goes in the prompt.
  4. The same question on /inference/chat/completions.
  5. The same question on /inference/responses.

Fusion reasons on every text-only request (/no_think only takes effect when
the request carries an image or video), so the helpers give it a 4096-token
budget and a mild repetition penalty, and strip the <think> block. Expect
15-40 s per text call on Fusion versus 1-3 s on C 2.6.

Usage:
    cp .env.example .env  # then fill in ATAI_API_KEY and ATAI_API_ENDPOINT
    python text_query.py                  # Newton-C 2.6
    python text_query.py --model fusion   # Newton-Fusion 1.0

Data attribution:
    The flow log in example 3 is real data from the GHOST-IoT public dataset
    (WiFi flows captured on the wlan0 interface of a real smart-home gateway;
    MAC addresses anonymized in the source). Examples 1 and 2 use synthetic
    flows in the same format.
      Source: GHOST-IoT dataset, EU Horizon 2020 project
              https://github.com/gspathoulas/ghost-iot-dataset
      Paper:  Anagnostopoulos, M.; Spathoulas, G.; Viaño, B.;
              Augusto-Gonzalez, J. "Tracing Your Smart-Home Devices
              Conversations: A Real World IoT Traffic Data-Set."
              Sensors 2020, 20, 6600. https://doi.org/10.3390/s20226600
    See the source repository for the dataset's license terms.
"""

from __future__ import annotations

import argparse
import json

from archetypeai.api_client import ArchetypeAI

from _common import banner, chat, make_client, model_arg, query, resolve_model, respond

ANALYST = "You are a network security analyst reviewing smart-home WiFi traffic."

# The pipe-separated flow-log format used throughout the GHOST-IoT pipeline.
FIELD_LEGEND = (
    "Flow log fields (pipe-separated): "
    "time_utc|mac_a|mac_b|prot|tran|port_a|port_b|bytes_a|bytes_b|pkts_a|pkts_b. "
    "Transport: 6=TCP, 17=UDP, 1=ICMP."
)


def example_plain_text(client: ArchetypeAI, **model_kw) -> None:
    banner("1. Plain text Q&A — interpret an inline WiFi flow snippet")
    text, _, elapsed_ms = query(
        client,
        user_query=(
            f"{FIELD_LEGEND}\n"
            "2019-10-19T08:14:22Z|aa1101000001|aa1100000001|DNS|17|53|56004|39|39|1|1\n"
            "2019-10-19T08:14:23Z|aa1101000001|aa1100000001|IMAPS|6|993|52650|2823|16030102|12|517\n"
            "2019-10-19T08:15:01Z|aa1102000007|aa1100000001|HTTPS|6|443|44120|18044|2204847|420|1580\n"
            "In 2-3 sentences, summarize what these devices were most likely doing."
        ),
        system_prompt=f"{ANALYST} Reply in 2-3 sentences. No preamble, no caveats.",
        max_new_tokens=256,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_json_output(client: ArchetypeAI, **model_kw) -> None:
    banner("2. Structured JSON output — classify a device (the prompt is the schema)")
    text, _, elapsed_ms = query(
        client,
        user_query=(
            "Classify the smart-home device with this one-day flow profile: "
            "dominant_protocols=[MQTT, NTP, DNS], total_bytes_up=1.4MB, "
            "total_bytes_down=0.2MB, flow_count=512, mean_flow_bytes=3100, "
            "active_hours=24, distinct_peers=2. "
            "(Note: total_bytes_up + total_bytes_down ~= flow_count * mean_flow_bytes.) "
            "Respond with ONLY the JSON object, no markdown fences, no prose."
        ),
        system_prompt=(
            f"{ANALYST} You output a single JSON object with this shape:\n"
            '{"reasoning": "<step-by-step interpretation of the protocols, '
            'volumes, and activity pattern>", '
            '"device_type": "phone|laptop|smart_speaker|iot_sensor|camera|unknown", '
            '"confidence": <0..1 float>}\n'
            "Reason in the `reasoning` field first, then commit to a "
            "`device_type`. Output only the JSON object."
        ),
        max_new_tokens=400,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")
    try:
        parsed = json.loads(text)
        print(
            f"Parsed JSON: device_type={parsed.get('device_type')!r} "
            f"confidence={parsed.get('confidence')!r}\n"
        )
    except json.JSONDecodeError as decode_error:
        print(f"WARN: model output did not parse as JSON: {decode_error}\n")


# Real GHOST-IoT flows: device e323b826aa71 over a ~100-second window on
# 2019-10-19 (values verbatim from data/wlan0_ipv4_flows_db.csv; the source
# HH:MM:SS times are shown as full ISO timestamps for consistency with the
# other examples). The device joins via DHCP, resolves DNS, then opens a burst
# of HTTP/HTTPS connections — a web-browsing session. It appears as mac_a in
# the DHCP flow and mac_b in the rest.
FLOW_LOG = (
    f"{FIELD_LEGEND}\n"
    "2019-10-19T15:55:11Z|ebd1a7fa8544|e323b826aa71|DHCP|17|67|68|0|900|0|3\n"
    "2019-10-19T15:55:14Z|e323b826aa71|13d35af5c06b|DHCP|17|68|67|900|0|3|0\n"
    "2019-10-19T15:55:17Z|13d35af5c06b|e323b826aa71|DNS|17|53|55356|200|72|2|2\n"
    "2019-10-19T15:55:17Z|13d35af5c06b|e323b826aa71|DNS|17|53|56188|200|72|2|2\n"
    "2019-10-19T15:55:17Z|13d35af5c06b|e323b826aa71|Unknown_TCP|6|443|65438|66|31|5|5\n"
    "2019-10-19T15:55:18Z|13d35af5c06b|e323b826aa71|HTTPS|6|443|63160|5074|1079|11|15\n"
    "2019-10-19T15:55:20Z|13d35af5c06b|e323b826aa71|HTTPS|6|443|63161|14846|1205|16|16\n"
    "2019-10-19T15:55:22Z|13d35af5c06b|e323b826aa71|HTTP|6|80|63165|370217|814|278|189\n"
    "2019-10-19T15:55:22Z|13d35af5c06b|e323b826aa71|HTTP|6|80|63166|641705|1879|472|334\n"
    "2019-10-19T15:55:24Z|13d35af5c06b|e323b826aa71|HTTPS|6|443|63169|4568|18001|22|28\n"
    "2019-10-19T15:55:40Z|13d35af5c06b|e323b826aa71|HTTP|6|80|63172|67978|1054|53|42\n"
    "2019-10-19T15:56:18Z|13d35af5c06b|e323b826aa71|HTTP|6|80|63183|67773|654|52|38\n"
    "2019-10-19T15:56:35Z|13d35af5c06b|e323b826aa71|HTTP|6|80|63189|99258|1154|74|75\n"
    "2019-10-19T15:56:51Z|13d35af5c06b|e323b826aa71|HTTPS|6|443|63195|6169|1360|13|18\n"
)


DEVICE_QUESTION = (
    "Analyze the flow log below for a single device (e323b826aa71) and "
    "describe what it did during this capture window. Summarize its activity "
    "level and protocol mix, infer what kind of device it likely is, and flag "
    "anything unusual. Reply in 2-3 sentences, no preamble.\n\n"
)
DEVICE_SYSTEM = (
    f"{ANALYST} The device under review (e323b826aa71) appears as endpoint a "
    "or b in every flow."
)


def example_flow_log_inline(client: ArchetypeAI, **model_kw) -> None:
    banner("3. Reasoning over a longer flow log, inlined in the query")
    text, _, elapsed_ms = query(
        client,
        user_query=DEVICE_QUESTION + FLOW_LOG,
        system_prompt=DEVICE_SYSTEM,
        max_new_tokens=400,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_chat_completions(client: ArchetypeAI, **model_kw) -> None:
    banner("4. Same question on /inference/chat/completions")
    text, _, elapsed_ms = chat(
        client,
        [{"type": "text", "text": DEVICE_QUESTION + FLOW_LOG}],
        system_prompt=DEVICE_SYSTEM,
        max_tokens=400,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def example_responses(client: ArchetypeAI, **model_kw) -> None:
    banner("5. Same question on /inference/responses")
    text, _, elapsed_ms = respond(
        client,
        [{"type": "input_text", "text": DEVICE_QUESTION + FLOW_LOG}],
        instructions=DEVICE_SYSTEM,
        max_output_tokens=400,
        **model_kw,
    )
    print(f"[{elapsed_ms} ms]\n{text}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    model_arg(parser)
    args = parser.parse_args()
    model_kw = {"model": resolve_model(args.model), "reasoning": args.reasoning}

    client = make_client()
    print(f"model={model_kw['model']} reasoning={args.reasoning}\n")
    example_plain_text(client, **model_kw)
    example_json_output(client, **model_kw)
    example_flow_log_inline(client, **model_kw)
    example_chat_completions(client, **model_kw)
    example_responses(client, **model_kw)


if __name__ == "__main__":
    main()
