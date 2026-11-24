"""Plot observed training metrics; no smoothing or inference of policy quality."""
import argparse
import csv
import json
from pathlib import Path


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    episodes = read_jsonl(args.run/'training_episodes.jsonl')
    metrics = read_jsonl(args.run/'learning_metrics.jsonl')
    resets = read_jsonl(args.run/'training_resets.jsonl')
    if len(resets) < len(episodes):
        raise ValueError("Missing reset records for training episodes")
    for episode, reset in zip(episodes, resets):
        episode['simulator_seed'] = reset['episode_seed']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for seed in sorted({e['simulator_seed'] for e in episodes}):
        group = [e for e in episodes if e['simulator_seed'] == seed]
        steps = [e['steps'] for e in group]
        axes[0, 0].plot(steps, [e['episode']['r'] for e in group], 'o-', label=f"sim {seed}")
        axes[0, 1].plot(steps, [e['episode'].get('task_error', e['episode'].get('door_openness')) for e in group], 'o-', label=f"sim {seed}")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    axes[0, 0].set(ylabel='Episode shaped return', title='Training episodes (exploration enabled)')
    axes[0, 1].set(ylabel='Final task error (CloseFridge: mean openness)', title='No held-out inference from this curve')
    for ax, field, label in [(axes[1, 0], 'train/actor_loss', 'Actor loss'), (axes[1, 1], 'train/critic_loss', 'Critic loss')]:
        values = [m for m in metrics if field in m]
        ax.plot([m['steps'] for m in values], [m[field] for m in values])
        ax.set(ylabel=label)
    for ax in axes.flat:
        ax.set_xlabel('Environment steps')
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.run/'learning_curves.png', dpi=150)
    plt.close(fig)
    summary = {'episodes': len(episodes), 'training_successes': sum(e['success'] for e in episodes),
               'environment_steps': metrics[-1]['steps'], 'optimizer_updates': metrics[-1]['updates'],
               'elapsed_s': metrics[-1]['elapsed_s'],
               'per_initialization': {str(seed): {'episodes': sum(e['simulator_seed'] == seed for e in episodes),
                    'successes': sum(e['success'] and e['simulator_seed'] == seed for e in episodes)}
                    for seed in sorted({e['simulator_seed'] for e in episodes})},
               'note': 'Training results with exploration; not held-out performance.'}
    (args.run/'learning_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
