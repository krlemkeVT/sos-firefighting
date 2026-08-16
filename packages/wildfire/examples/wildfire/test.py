from examples.wildfire.main import run_sim

p = run_sim(
    input_file="Palisades.json",
    overwrites_file="baseline_palisades.json",
    seed=0,
    force_headless=True,
)

print("Output base path:", p)
print("JSON exists?", p.with_suffix(".json").exists())
