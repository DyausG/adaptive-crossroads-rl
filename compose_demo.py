"""Annotate and combine genuine GUI screenshots into a matched recorded replay."""
import argparse
import json
from PIL import Image, ImageDraw, ImageFont
from runtime import ROOT
from experiments import get_experiment


def main(experiment="v2", write=None):
    exp = get_experiment(experiment)
    if write is None:
        write = (exp.experiment_id != "v1")
    if not write:
        print("v1 demo artifacts are frozen; pass --write to compose v1 demo explicitly.")
        return

    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 23)
    small = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    metadata = [json.loads((exp.demo_dir / f"raw/{c}_seed0/capture.json").read_text(encoding="utf-8")) for c in ["rule", "emergency"]]
    assert metadata[0]["route_sha256"] == metadata[1]["route_sha256"]
    assert all(x["headless_metrics_match"] for x in metadata)
    scenario_id = metadata[0]["scenario"]
    output = exp.demo_dir / "frames"
    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for left, right in zip(metadata[0]["frames"], metadata[1]["frames"]):
        assert left["requested_at_sim_second"] == right["requested_at_sim_second"]
        t = left["requested_at_sim_second"]
        frame = Image.new("RGB", (1440, 775), (18, 29, 44))
        draw = ImageDraw.Draw(frame)
        draw.text((20, 12), "Rule-based preemption", font=font, fill=(221, 237, 248))
        draw.text((740, 12), f"Emergency-aware DQN ({exp.experiment_id}) | seed 0", font=font, fill=(221, 237, 248))
        p_left = metadata[0]["metrics"]["ambulance_passage_seconds"]
        p_right = metadata[1]["metrics"]["ambulance_passage_seconds"]
        draw.text((20, 46), f"Full-run passage: {p_left:.0f} s", font=small, fill=(111, 218, 196))
        draw.text((740, 46), f"Full-run passage: {p_right:.0f} s", font=small, fill=(246, 172, 128))
        frame.paste(Image.open(ROOT / left["path"]).convert("RGB"), (10, 78))
        frame.paste(Image.open(ROOT / right["path"]).convert("RGB"), (730, 78))
        draw.text((20, 731), f"Same rush-hour route: {scenario_id} | screenshot requested at t={t} s; rendered on next step", font=small, fill=(221, 237, 248))
        draw.text((20, 752), f"Genuine SUMO-GUI recording ({exp.experiment_id}) | 20 simulated seconds between frames | predeclared seed 0 case", font=small, fill=(156, 181, 202))
        frame.save(output / f"comparison_{t:04d}.png")
        frames.append(frame)
    frames[0].save(exp.demo_dir / "comparison.gif", save_all=True, append_images=frames[1:], duration=600, loop=0, optimize=True)
    Image.open(output / "comparison_0400.png").save(exp.demo_dir / "comparison.png")
    (exp.demo_dir / "README.md").write_text(
        f"# Matched recorded comparison ({exp.experiment_id})\n\n"
        f"Genuine SUMO-GUI screenshots on {scenario_id}: the predeclared held-out asymmetric rush-hour ambulance scenario. "
        f"Rule preemption ({p_left:.0f} s) versus emergency-aware model seed 0 ({p_right:.0f} s). "
        "Both GUI replays exactly matched headless passage time, regular waiting and unfinished/uninserted counts.\n\n"
        f"{len(frames)} paired frames, requested at simulation times 200 through 900s in 20s intervals. "
        "SUMO saves a requested screenshot on the following simulation step. GIF playback is a sampled time-compressed recording, "
        "not a live simulation. Original PNGs, screenshot hashes and controller metrics are in raw/*/capture.json.\n",
        encoding="utf-8",
    )
    print(f"Composed {len(frames)} matched frames and sampled animated replay for {exp.experiment_id} in {exp.demo_dir}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["v1", "v2"], default="v2")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    main(args.experiment, write=True if args.write else None)
