"""Synthetic Agent, Tool-Calling, and Reasoning Data Generator.

Generates rich, high-diversity training trajectories for frontier-grade agent behavior:
1. Multi-hop Web Search & Fact Retrieval (web_search) with reasoning (<thought>...</thought>).
2. Python Interpreter / Execution (python_interpreter) for exact calculations and verification.
3. Multi-turn ReAct chains (thought -> action -> observation -> thought -> action -> final response).
4. Contrastive Negative Samples (tools declared in system prompt, but model correctly reasons
   that external tools are unnecessary and answers directly from internal knowledge).

Produces standard OpenAI-compatible JSONL format (tools + messages) ready for
direct ingestion into DrunkenBot LLM-IDE with sequence packing and prompt loss masking.

Usage:
    python -m engine.generate_agent_data [output_path] \
        [--samples 500] \
        [--contrastive-ratio 0.3] \
        [--multihop-ratio 0.4]
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Optional

from .tool_call_data import STANDARD_AGENT_TOOLS, create_contrastive_negative_sample, create_multihop_agent_trajectory


# ---------------------------------------------------------------------------
# Synthetic Agent Knowledge & Scenario Banks
# ---------------------------------------------------------------------------

SEARCH_SCENARIOS = [
    {
        "topic": "James Webb Space Telescope",
        "query": "What are the latest findings from the James Webb Space Telescope regarding galaxy GN-z11?",
        "thought_search": "The user is asking about recent astronomical observations of GN-z11 made by JWST. I need to search the web for recent spectral analysis and redshift confirmations.",
        "search_term": "JWST discoveries galaxy GN-z11 redshift spectra",
        "search_result": json.dumps({
            "results": [
                {
                    "title": "NASA JWST Confirms Redshift of GN-z11",
                    "snippet": "JWST NIRSpec observations confirmed GN-z11 at redshift z=10.6, existing just 430 million years after the Big Bang, with evidence of a supermassive black hole accreting matter.",
                }
            ]
        }),
        "thought_final": "The search result provides the redshift confirmation (z=10.6), the era (430 million years after the Big Bang), and detection of an actively accreting supermassive black hole. I can now synthesize a clear, informative answer.",
        "answer": "Observations from the James Webb Space Telescope (JWST) using its NIRSpec instrument confirmed that galaxy GN-z11 has a spectroscopic redshift of z = 10.6. This places it approximately 430 million years after the Big Bang. Additionally, spectral signatures revealed ionised neon and nitrogen, indicating the presence of an actively accreting supermassive black hole in the early universe.",
    },
    {
        "topic": "Perseverance Rover on Mars",
        "query": "What rock core samples did NASA's Perseverance rover collect in Jezero Crater recently?",
        "thought_search": "I need current information about the specific rock core samples cached by Perseverance in Jezero Crater.",
        "search_term": "NASA Perseverance rover Jezero Crater rock core samples cached",
        "search_result": json.dumps({
            "results": [
                {
                    "title": "Perseverance Rover Science Cache Update",
                    "snippet": "Perseverance collected sedimentary and igneous rock cores from the 'Skinner Ridge' and 'Wildcat Ridge' formations in the delta front, rich in sulfates and carbonates that could preserve biosignatures.",
                }
            ]
        }),
        "thought_final": "The observation identifies key formations ('Skinner Ridge' and 'Wildcat Ridge') containing sedimentary deltaic rocks and carbonates. I will answer the user's question clearly.",
        "answer": "NASA's Perseverance rover has collected several rock core samples from Jezero Crater, notably from formations including 'Skinner Ridge' and 'Wildcat Ridge' along the ancient river delta. These fine-grained sedimentary samples are rich in carbonates and sulfates, minerals with high potential to preserve microscopic biosignatures from Mars' ancient wet era.",
    },
    {
        "topic": "Global Renewable Energy Milestone",
        "query": "What percentage of global electricity was generated from solar and wind in 2024?",
        "thought_search": "I need to look up authoritative statistical reports (such as Ember or IEA) regarding the share of wind and solar in global electricity generation for 2024.",
        "search_term": "global electricity generation share solar wind 2024 Ember IEA report",
        "search_result": json.dumps({
            "results": [
                {
                    "title": "Global Electricity Review 2024",
                    "snippet": "Wind and solar reached a historic record of 15% of global electricity generation in 2024, up from 13.4% in 2023, driving renewables overall past 30% of the world's power.",
                }
            ]
        }),
        "thought_final": "The data from the global review reports wind and solar reaching 15% in 2024, with total renewables exceeding 30%. I will present this exact breakdown.",
        "answer": "According to energy analytics reports from Ember and the IEA, solar and wind generated approximately 15% of global electricity in 2024, setting an all-time record. When combined with hydroelectric, geothermal, and other renewable sources, clean electricity surpassed 30% of global generation.",
    },
    {
        "topic": "Voyager 1 Communications",
        "query": "How did NASA engineers resolve the telecommunications issue with Voyager 1 in 2024?",
        "thought_search": "Voyager 1 experienced memory corruption in its Flight Data System (FDS) in 2024. Let me search for the specific engineering fix used by JPL.",
        "search_term": "Voyager 1 flight data system FDS memory chip repair JPL 2024",
        "search_result": json.dumps({
            "results": [
                {
                    "title": "JPL Restores Voyager 1 Science Data",
                    "snippet": "Engineers traced the malfunction to a single corrupted chip holding 3% of the FDS memory. Because a hardware fix was impossible at 24 billion km, JPL relocated the affected code in sections across remaining memory addresses.",
                }
            ]
        }),
        "thought_final": "The solution was identifying a single failed memory chip in the FDS and relocating the affected software routines across available memory sectors over 24 billion kilometers away. I will provide a detailed summary.",
        "answer": "In 2024, NASA Jet Propulsion Laboratory (JPL) engineers resolved Voyager 1's telemetry issue by determining that a single failed memory chip (representing about 3% of the Flight Data System's memory) had corrupted the data formatting routines. Because physical replacement was impossible at a distance of over 24 billion kilometers, the team cleverly divided the corrupted code into smaller segments and relocated them into unused sectors throughout the remaining FDS memory, successfully restoring science transmissions.",
    },
    {
        "topic": "Superconductivity Research",
        "query": "What is the status of ambient-pressure room-temperature superconductivity claims?",
        "thought_search": "The user is inquiring about the scientific consensus on claims of room-temperature, ambient-pressure superconductivity (such as LK-99 and subsequent materials).",
        "search_term": "room temperature ambient pressure superconductor replication scientific consensus",
        "search_result": json.dumps({
            "results": [
                {
                    "title": "International Consensus on LK-99 and Ambient Superconductivity",
                    "snippet": "Rigorous replication efforts by global laboratories (Max Planck, Princeton, CSIR) determined that apparent zero-resistance and levitation in LK-99 were due to copper sulfide (Cu2S) phase transitions and ferromagnetism, not superconductivity.",
                }
            ]
        }),
        "thought_final": "The peer-reviewed international consensus is that ambient superconductivity was not confirmed, with anomalies attributed to Cu2S impurities. I will present this balanced, evidence-based status.",
        "answer": "To date, independent international replication efforts by research institutions—including the Max Planck Institute, Princeton University, and others—have not validated any claims of ambient-pressure room-temperature superconductivity. Investigations revealed that phenomena like partial levitation and resistance drops in materials such as LK-99 were caused by ferromagnetism and phase transitions in copper(I) sulfide (Cu₂S) impurities rather than true superconductivity.",
    },
]

PYTHON_SCENARIOS = [
    {
        "topic": "Orbital Velocity Calculation",
        "query": "Calculate the orbital speed of the International Space Station orbiting Earth at an altitude of 420 km. Assume Earth radius = 6371 km and GM = 3.986e14 m^3/s^2.",
        "thought_code": "The orbital speed formula is v = sqrt(GM / r), where r = Earth radius + altitude. Let me write a Python script to calculate this precisely.",
        "code": "import math\n\ngm = 3.986e14  # m^3/s^2\nearth_r = 6371 * 1000  # meters\naltitude = 420 * 1000  # meters\nr = earth_r + altitude\nv = math.sqrt(gm / r)\nprint(f'{v:.2f}')",
        "code_result": "7660.03",
        "thought_final": "The calculated speed is 7660.03 m/s, or approximately 7.66 km/s (about 27,576 km/h). I will state the result with both metric and common speed units.",
        "answer": "The orbital velocity of the International Space Station at an altitude of 420 km is approximately **7,660 m/s** (7.66 km/s or about 27,576 km/h / 17,135 mph).\n\nFormula used: $v = \\sqrt{\\frac{GM}{R + h}} = \\sqrt{\\frac{3.986 \\times 10^{14}}{6,371,000 + 420,000}} \\approx 7660.03\\text{ m/s}$.",
    },
    {
        "topic": "Compound Interest & Financial Planning",
        "query": "If I invest $15,000 at an annual interest rate of 7.5% compounded monthly, how much will I have after 12 years?",
        "thought_code": "Compound interest formula: A = P * (1 + r/n)^(n*t), with P=15000, r=0.075, n=12, t=12. Let me evaluate this in Python.",
        "code": "p = 15000\nr = 0.075\nn = 12\nt = 12\na = p * ((1 + r/n) ** (n * t))\ninterest = a - p\nprint(f'Total: {a:.2f}, Interest: {interest:.2f}')",
        "code_result": "Total: 36767.14, Interest: 21767.14",
        "thought_final": "The final amount is $36,767.14, representing $21,767.14 in earned compound interest. I will explain the calculation step by step.",
        "answer": "After 12 years of monthly compounding at 7.5% annual interest, your investment of $15,000 will grow to **$36,767.14**.\n\n- **Principal Invested**: $15,000.00\n- **Total Interest Earned**: $21,767.14\n- **Final Balance**: $36,767.14\n\nFormula: $A = P \\left(1 + \\frac{r}{n}\\right)^{nt} = 15000 \\left(1 + \\frac{0.075}{12}\\right)^{144} \\approx \\$36,767.14$.",
    },
    {
        "topic": "Statistical Standard Deviation",
        "query": "Given the test scores [88, 92, 79, 95, 84, 91, 76, 89, 94, 82], calculate the mean and sample standard deviation.",
        "thought_code": "I need to calculate the sample mean and sample standard deviation (with Bessel's correction n-1) using Python.",
        "code": "import statistics\nscores = [88, 92, 79, 95, 84, 91, 76, 89, 94, 82]\nmean_val = statistics.mean(scores)\nstdev_val = statistics.stdev(scores)\nprint(f'Mean: {mean_val:.2f}, Stdev: {stdev_val:.2f}')",
        "code_result": "Mean: 87.00, Stdev: 6.27",
        "thought_final": "The sample mean is 87.00 and sample standard deviation is 6.27. I will summarize the findings clearly.",
        "answer": "For the test scores provided:\n- **Mean score**: 87.00\n- **Sample Standard Deviation ($s$)**: 6.27\n\nThe scores exhibit a tight distribution around 87, with all values falling within approximately 1.8 standard deviations of the mean.",
    },
    {
        "topic": "Kinetic Energy Calculation",
        "query": "What is the kinetic energy in megajoules of a 1,200 kg car traveling at 110 km/h?",
        "thought_code": "Kinetic energy KE = 0.5 * m * v^2. First convert 110 km/h to m/s (divide by 3.6), compute KE in joules, then divide by 1e6 for megajoules.",
        "code": "m = 1200  # kg\nv_kmh = 110\nv_ms = v_kmh / 3.6\nke_j = 0.5 * m * (v_ms ** 2)\nke_mj = ke_j / 1e6\nprint(f'{ke_mj:.3f} MJ')",
        "code_result": "0.560 MJ",
        "thought_final": "110 km/h is ~30.56 m/s. 0.5 * 1200 * (30.56)^2 = 560,185 J = 0.560 MJ. I will present the complete derivation.",
        "answer": "A 1,200 kg vehicle traveling at 110 km/h possesses approximately **0.560 Megajoules (MJ)** (or 560.19 kJ) of kinetic energy.\n\n**Calculation**:\n1. Speed conversion: $110\\text{ km/h} \\div 3.6 = 30.556\\text{ m/s}$\n2. Kinetic Energy: $KE = \\frac{1}{2}mv^2 = \\frac{1}{2} \\times 1200 \\times (30.556)^2 = 560,185\\text{ Joules} = 0.560\\text{ MJ}$.",
    },
]

MULTIHOP_SCENARIOS = [
    {
        "topic": "Mars Distance & Radio Communication Delay",
        "query": "What is the minimum distance between Earth and Mars, and how long does it take for a radio signal to travel that distance in minutes?",
        "thought_1": "I need to find the closest recorded distance between Earth and Mars (perihelic opposition). Let me search the web.",
        "tool_1_name": "web_search",
        "tool_1_args": {"query": "Earth Mars closest approach distance kilometers"},
        "result_1": json.dumps({"results": [{"snippet": "The closest recorded approach between Earth and Mars was in August 2003 at approximately 54.6 million kilometers (0.364 AU)."}]}),
        "thought_2": "The search gives the distance as 54.6 million kilometers (54,600,000 km). Now I need to compute the radio light travel time using Python with speed of light c = 299,792 km/s.",
        "tool_2_name": "python_interpreter",
        "tool_2_args": {"code": "dist_km = 54_600_000\nc_km_s = 299_792.458\nseconds = dist_km / c_km_s\nminutes = seconds / 60\nprint(f'{seconds:.2f} seconds ({minutes:.2f} minutes)')"},
        "result_2": "182.13 seconds (3.04 minutes)",
        "thought_3": "I now have both the physical distance from search and the precise radio communication latency computed via Python. I can assemble the final comprehensive answer.",
        "answer": "At their closest recorded approach (such as in August 2003), Earth and Mars were approximately **54.6 million kilometers** (33.9 million miles or 0.364 AU) apart.\n\nRadio signals travel at the speed of light ($c \\approx 299,792\\text{ km/s}$). Over this minimum distance, a one-way communication signal takes:\n- **182.13 seconds**, or approximately **3.04 minutes** (3 minutes and 2 seconds).\n\nDuring distant alignments on opposite sides of the Sun (superior conjunction), this delay can expand to over 22 minutes one-way.",
    },
    {
        "topic": "Country Population Density Comparison",
        "query": "Look up the land area and population of Iceland, and calculate its population density per square kilometer.",
        "thought_1": "I need to look up official demographic and geographic figures for Iceland.",
        "tool_1_name": "web_search",
        "tool_1_args": {"query": "Iceland population land area square kilometers official statistics"},
        "result_1": json.dumps({"results": [{"snippet": "According to Statistics Iceland, the population is approximately 398,000 with a total land area of 103,000 square kilometers."}]}),
        "thought_2": "Now I have population = 398,000 and area = 103,000 km^2. Let me use Python to calculate the exact population density.",
        "tool_2_name": "python_interpreter",
        "tool_2_args": {"code": "population = 398_000\narea_km2 = 103_000\ndensity = population / area_km2\nprint(f'{density:.2f} people/km2')"},
        "result_2": "3.86 people/km2",
        "thought_3": "The density is 3.86 people per km^2. Iceland is one of the most sparsely populated countries in Europe. I will summarize this clearly.",
        "answer": "Based on demographic records:\n- **Population of Iceland**: ~398,000 residents\n- **Total Land Area**: ~103,000 km²\n\n**Population Density Calculation**:\n$$\\text{Density} = \\frac{398,000}{103,000} \\approx \\mathbf{3.86\\text{ people per square kilometer}}$$\n\nThis makes Iceland the most sparsely populated country in Europe.",
    },
]

CONTRASTIVE_NEGATIVE_SCENARIOS = [
    {
        "query": "What is the capital of Australia, and what is its official language?",
        "thought": "The user is asking for the capital and official language of Australia. This is established geographic knowledge that does not require external search or code execution. The capital is Canberra, and the de facto official language is English. I will answer directly without tools.",
        "answer": "The capital of Australia is **Canberra**. The country's de facto national and official language is **English**.",
    },
    {
        "query": "Explain how photosynthesis works in green plants.",
        "thought": "The query asks for a conceptual explanation of photosynthesis. This is fundamental biological science within my internal knowledge. No external tools or calculations are required. I will structure the explanation covering the light-dependent reactions and the Calvin cycle.",
        "answer": "Photosynthesis is the biological process by which green plants, algae, and certain bacteria convert sunlight into chemical energy. It occurs primarily within the chloroplasts of plant cells:\n\n1. **Light-Dependent Reactions** (Thylakoids): Chlorophyll absorbs light energy, splitting water molecules ($H_2O$) into oxygen ($O_2$), protons, and electrons, generating ATP and NADPH.\n2. **Calvin Cycle / Light-Independent Reactions** (Stroma): ATP and NADPH power the fixation of carbon dioxide ($CO_2$) into glucose ($C_6H_{12}O_6$).\n\n**Overall Equation**:\n$$6CO_2 + 6H_2O + \\text{light energy} \\longrightarrow C_6H_{12}O_6 + 6O_2$$",
    },
    {
        "query": "What is the difference between a stack and a queue data structure?",
        "thought": "This is a standard computer science question about fundamental data structures. No external web search or calculation is needed. I will explain LIFO vs FIFO semantics with common use cases.",
        "answer": "The primary difference between a **stack** and a **queue** lies in the order elements are inserted and removed:\n\n- **Stack (LIFO - Last In, First Out)**: The most recently added element is the first to be removed (like a stack of cafeteria plates). Key operations are `push` and `pop`. Used in call stack execution, undo mechanisms, and depth-first search.\n- **Queue (FIFO - First In, First Out)**: The earliest added element is the first to be removed (like people waiting in line). Key operations are `enqueue` and `dequeue`. Used in task scheduling, breadth-first search, and IO buffer management.",
    },
    {
        "query": "Write a welcoming greeting for a software engineering conference.",
        "thought": "This is a creative writing request for a conference welcome message. Internal language capabilities are fully sufficient; external tools are not required.",
        "answer": "Welcome, engineers, architects, and innovators, to DevSummit 2026! Over the next three days, we come together to explore breakthrough technologies, share engineering lessons, and collaborate on building scalable, reliable systems for the future. Dive into sessions, connect with fellow builders, and welcome to the community!",
    },
    {
        "query": "Who wrote 'The Republic' and what is its central philosophical allegory?",
        "thought": "This asks about classical philosophy regarding Plato's Republic. Established historical knowledge covers this completely without external tools.",
        "answer": "**The Republic** was written by the classical Greek philosopher **Plato** around 375 BCE. Its central and most famous allegory is the **Allegory of the Cave** (found in Book VII), which illustrates the philosophical journey from perceptual illusion to true knowledge and enlightenment.",
    },
]


# ---------------------------------------------------------------------------
# Generator Functions
# ---------------------------------------------------------------------------

def create_search_sample(scenario: dict[str, str]) -> dict[str, Any]:
    """Build an OpenAI-compatible agent search trajectory with <thought> reasoning."""
    call_id = f"call_{random.randint(10000, 99999)}"
    return {
        "tools": STANDARD_AGENT_TOOLS,
        "messages": [
            {"role": "user", "content": scenario["query"]},
            {
                "role": "assistant",
                "content": f"<thought>{scenario['thought_search']}</thought>",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": scenario["search_term"]}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": scenario["search_result"],
            },
            {
                "role": "assistant",
                "content": f"<thought>{scenario['thought_final']}</thought>\n{scenario['answer']}",
            },
        ],
    }


def create_python_sample(scenario: dict[str, str]) -> dict[str, Any]:
    """Build an OpenAI-compatible Python interpreter trajectory with <thought> reasoning."""
    call_id = f"call_{random.randint(10000, 99999)}"
    return {
        "tools": STANDARD_AGENT_TOOLS,
        "messages": [
            {"role": "user", "content": scenario["query"]},
            {
                "role": "assistant",
                "content": f"<thought>{scenario['thought_code']}</thought>",
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": "python_interpreter",
                            "arguments": json.dumps({"code": scenario["code"]}),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id,
                "content": scenario["code_result"],
            },
            {
                "role": "assistant",
                "content": f"<thought>{scenario['thought_final']}</thought>\n{scenario['answer']}",
            },
        ],
    }


def create_multihop_sample(scenario: dict[str, Any]) -> dict[str, Any]:
    """Build a multi-hop trajectory combining search, python calculation, and multi-turn reasoning."""
    call_id_1 = f"call_{random.randint(10000, 99999)}"
    call_id_2 = f"call_{random.randint(10000, 99999)}"
    return {
        "tools": STANDARD_AGENT_TOOLS,
        "messages": [
            {"role": "user", "content": scenario["query"]},
            {
                "role": "assistant",
                "content": f"<thought>{scenario['thought_1']}</thought>",
                "tool_calls": [
                    {
                        "id": call_id_1,
                        "type": "function",
                        "function": {
                            "name": scenario["tool_1_name"],
                            "arguments": json.dumps(scenario["tool_1_args"]),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id_1,
                "content": scenario["result_1"],
            },
            {
                "role": "assistant",
                "content": f"<thought>{scenario['thought_2']}</thought>",
                "tool_calls": [
                    {
                        "id": call_id_2,
                        "type": "function",
                        "function": {
                            "name": scenario["tool_2_name"],
                            "arguments": json.dumps(scenario["tool_2_args"]),
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": call_id_2,
                "content": scenario["result_2"],
            },
            {
                "role": "assistant",
                "content": f"<thought>{scenario['thought_3']}</thought>\n{scenario['answer']}",
            },
        ],
    }


def generate_agent_dataset(
    output_path: Path,
    sample_count: int = 200,
    contrastive_ratio: float = 0.30,
    multihop_ratio: float = 0.30,
    seed: int = 42,
) -> int:
    """Generate a high-quality synthetic agent & reasoning JSONL corpus file.

    Args:
        output_path: Path to output .jsonl file.
        sample_count: Total records to generate.
        contrastive_ratio: Fraction of negative contrastive samples (no tool calls).
        multihop_ratio: Fraction of multi-hop ReAct chains (multiple tools).
        seed: Random seed for reproducible generation.

    Returns:
        Number of records written.
    """
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written_count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for idx in range(sample_count):
            roll = rng.random()

            if roll < contrastive_ratio:
                # Contrastive negative sample: tools declared, thought explains why no tool is needed
                sc = rng.choice(CONTRASTIVE_NEGATIVE_SCENARIOS)
                record = {
                    "tools": STANDARD_AGENT_TOOLS,
                    "messages": [
                        {"role": "user", "content": sc["query"]},
                        {"role": "assistant", "content": f"<thought>{sc['thought']}</thought>\n{sc['answer']}"},
                    ],
                }
            elif roll < (contrastive_ratio + multihop_ratio):
                # Multi-hop trajectory
                sc = rng.choice(MULTIHOP_SCENARIOS)
                record = create_multihop_sample(sc)
            else:
                # Single-hop tool call: alternate search and python execution
                if rng.random() < 0.5:
                    sc = rng.choice(SEARCH_SCENARIOS)
                    record = create_search_sample(sc)
                else:
                    sc = rng.choice(PYTHON_SCENARIOS)
                    record = create_python_sample(sc)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            written_count += 1

    return written_count


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate synthetic agent & reasoning JSONL dataset.")
    parser.add_argument("positional_output", nargs="?", default=None, help="Output .jsonl path (positional)")
    parser.add_argument("-o", "--output", default="training_data/agent_trajectories.jsonl", help="Output .jsonl path")
    parser.add_argument("--samples", type=int, default=200, help="Number of records to generate")
    parser.add_argument("--contrastive-ratio", type=float, default=0.30, help="Ratio of contrastive negative samples")
    parser.add_argument("--multihop-ratio", type=float, default=0.30, help="Ratio of multi-hop trajectories")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    args = parser.parse_args()
    target_path = args.positional_output if args.positional_output is not None else args.output
    out = Path(target_path).resolve()
    count = generate_agent_dataset(
        output_path=out,
        sample_count=args.samples,
        contrastive_ratio=args.contrastive_ratio,
        multihop_ratio=args.multihop_ratio,
        seed=args.seed,
    )
    print(f"Successfully generated {count} synthetic agent records to: {out}")


if __name__ == "__main__":
    main()
